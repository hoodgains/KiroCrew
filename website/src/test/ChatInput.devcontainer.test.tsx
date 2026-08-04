import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, fireEvent, waitFor, act } from '@testing-library/react'
import { renderWithProviders } from './helpers'
import ChatInput from '../components/ChatInput'
import { api } from '../api/client'

// Pins the Dev Container chip's contract: it appears in the composer's context
// shelf ONLY while a container is actually up for the active project, it names
// the 12-char container id `docker ps` prints, and it owns the one exit from the
// container — a menu whose single item withdraws trust for this project.

const defaultProps = {
  value: '',
  onChange: vi.fn(),
  onSend: vi.fn(),
  onProjectClick: vi.fn(),
  project: '/home/u/work/KiroCrew',
}

const FULL_ID = '3f2a1b0c9d8e7f6a5b4c3d2e1f009988'

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

/** The chip's own control, addressed by its accessible name. */
const chipButton = () => screen.getByRole('button', { name: /dev container/i })

/** A ResizeObserver whose callback we can fire with a width of our choosing.
 *  The ambient stub in integration/setup.ts never fires, and happy-dom reports
 *  every contentRect as 0, so the shelf's compact breakpoint cannot be reached
 *  without driving the observer directly. Same harness as McpAppFrame.test.tsx. */
class FakeResizeObserver {
  static instances: FakeResizeObserver[] = []
  cb: ResizeObserverCallback
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb
    FakeResizeObserver.instances.push(this)
  }
  observe() {}
  unobserve() {}
  disconnect() {}
  fire(width: number) {
    this.cb(
      [{ contentRect: { width } }] as unknown as ResizeObserverEntry[],
      this as unknown as ResizeObserver,
    )
  }
}

/** Report `width` to every observer the render installed, so the shelf
 *  re-measures. Below 340px it drops the chips' label spans (shelfCompact). */
function resizeShelf(width: number) {
  act(() => {
    for (const ro of FakeResizeObserver.instances) ro.fire(width)
  })
}

describe('ChatInput Dev Container chip', () => {
  it('is absent while no container is running', () => {
    renderWithProviders(<ChatInput {...defaultProps} devcontainerId={FULL_ID} />)
    expect(screen.queryByText('Dev Container')).not.toBeInTheDocument()
  })

  it('names the container id truncated to twelve characters in the tooltip', () => {
    renderWithProviders(
      <ChatInput {...defaultProps} devcontainerRunning devcontainerId={FULL_ID} />,
    )
    const title = chipButton().getAttribute('title') || ''
    expect(title).toContain('3f2a1b0c9d8e')
    // The full 32-char id would overflow the tooltip and is not what the user
    // pastes into a docker command.
    expect(title).not.toContain(FULL_ID)
  })

  it('still reports the container when its id is unknown', () => {
    renderWithProviders(<ChatInput {...defaultProps} devcontainerRunning />)
    expect(chipButton().getAttribute('title')).toBe('This session runs in a Dev Container')
  })

  it('is a menu control, so it is reachable by keyboard', () => {
    renderWithProviders(
      <ChatInput {...defaultProps} devcontainerRunning devcontainerId={FULL_ID} />,
    )
    const chip = chipButton()
    expect(chip).toHaveAttribute('aria-haspopup', 'menu')
    expect(chip).toHaveAttribute('aria-expanded', 'false')
  })

  it('keeps the menu closed until the chip is clicked', () => {
    renderWithProviders(<ChatInput {...defaultProps} devcontainerRunning />)
    expect(screen.queryByRole('menuitem')).not.toBeInTheDocument()
    fireEvent.click(chipButton())
    expect(screen.getByRole('menuitem', { name: /withdraw trust/i })).toBeInTheDocument()
    expect(chipButton()).toHaveAttribute('aria-expanded', 'true')
  })

  it('withdraws trust for the status project, then refetches', async () => {
    // The trust key is the status response's realpath `project_dir`, not the
    // `project` label — a revoke against the label could miss the granted entry.
    const untrust = vi.spyOn(api, 'devcontainerUntrust').mockResolvedValue({
      trusted: false,
      removed: true,
    })
    const onDevcontainerUntrust = vi.fn().mockResolvedValue(undefined)
    renderWithProviders(
      <ChatInput
        {...defaultProps}
        devcontainerRunning
        devcontainerId={FULL_ID}
        devcontainerProject="/real/path/KiroCrew"
        onDevcontainerUntrust={onDevcontainerUntrust}
      />,
    )
    fireEvent.click(chipButton())
    fireEvent.click(screen.getByRole('menuitem', { name: /withdraw trust/i }))

    await waitFor(() => expect(untrust).toHaveBeenCalledWith('/real/path/KiroCrew'))
    await waitFor(() => expect(onDevcontainerUntrust).toHaveBeenCalledTimes(1))
    // Menu closes on success, so the chip does not look like it is still armed.
    await waitFor(() => expect(screen.queryByRole('menuitem')).not.toBeInTheDocument())
  })

  it('keeps the menu open and calls no refetch when the revoke fails', async () => {
    const untrust = vi
      .spyOn(api, 'devcontainerUntrust')
      .mockRejectedValue(new Error('gateway down'))
    const onDevcontainerUntrust = vi.fn()
    renderWithProviders(
      <ChatInput
        {...defaultProps}
        devcontainerRunning
        devcontainerProject="/real/path/KiroCrew"
        onDevcontainerUntrust={onDevcontainerUntrust}
      />,
    )
    fireEvent.click(chipButton())
    fireEvent.click(screen.getByRole('menuitem', { name: /withdraw trust/i }))

    await waitFor(() => expect(untrust).toHaveBeenCalledTimes(1))
    expect(onDevcontainerUntrust).not.toHaveBeenCalled()
    expect(screen.getByRole('menuitem', { name: /withdraw trust/i })).toBeInTheDocument()
  })

  it('disables the item with no project to revoke against', () => {
    renderWithProviders(<ChatInput {...defaultProps} devcontainerRunning />)
    fireEvent.click(chipButton())
    expect(screen.getByRole('menuitem', { name: /withdraw trust/i })).toBeDisabled()
  })

  // The chip is an icon + a label span that the compact shelf drops. Its name
  // therefore has to come from aria-label, not from the span, or a screen reader
  // reaches an unnamed menu button on a narrow window.
  describe('accessible name', () => {
    const origRO = globalThis.ResizeObserver

    beforeEach(() => {
      FakeResizeObserver.instances = []
      globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof ResizeObserver
    })
    afterEach(() => {
      globalThis.ResizeObserver = origRO
    })

    it('carries an explicit label rather than relying on the visible span', () => {
      renderWithProviders(
        <ChatInput {...defaultProps} devcontainerRunning devcontainerId={FULL_ID} />,
      )
      // Asserted on the attribute as well as the computed name: removing the
      // aria-label would still leave the wide-shelf name intact via the span,
      // so only the attribute pins the compact case's source of truth.
      expect(chipButton()).toHaveAttribute('aria-label', 'Dev Container')
    })

    it('keeps its name once the compact shelf drops the label span', () => {
      renderWithProviders(
        <ChatInput {...defaultProps} devcontainerRunning devcontainerId={FULL_ID} />,
      )
      expect(screen.getByText('Dev Container')).toBeInTheDocument()

      resizeShelf(200)

      // The span is gone and the Box icon is aria-hidden, so the name below can
      // only be coming from aria-label. Matched EXACTLY, not by substring: the
      // accname algorithm falls back to `title`, and the tooltip sentence
      // ("This session runs in a Dev Container…") also contains the phrase — a
      // loose match would pass with the aria-label deleted.
      expect(screen.queryByText('Dev Container')).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Dev Container' })).toBeInTheDocument()
    })

    it('drops the sibling chips to icon-only at the same width', () => {
      // Guards the test above from silently passing on a shelf that never went
      // compact: the agent and project chips lose their labels on the same flag.
      renderWithProviders(
        <ChatInput
          {...defaultProps}
          agentName="kirocrew"
          onAgentClick={vi.fn()}
          devcontainerRunning
        />,
      )
      expect(screen.getByText('kirocrew')).toBeInTheDocument()

      resizeShelf(200)

      expect(screen.queryByText('kirocrew')).not.toBeInTheDocument()
      // Still named, by the same aria-label pattern the Dev Container chip uses.
      expect(screen.getByRole('button', { name: /Agent: kirocrew/ })).toBeInTheDocument()
    })
  })
})
