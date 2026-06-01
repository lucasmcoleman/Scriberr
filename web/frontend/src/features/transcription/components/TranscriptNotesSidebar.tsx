import { useEffect, useState, type CSSProperties, type FormEvent, type KeyboardEvent, type PointerEvent } from "react";
import { GripHorizontal, MessageCircle, MessageSquareText, Mic2, Send, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TranscriptChatPanel } from "@/features/transcription/components/TranscriptChatPanel";
import type { TranscriptNoteAnnotation } from "@/features/transcription/api/annotationsApi";

type TranscriptNotesSidebarProps = {
  notes: TranscriptNoteAnnotation[];
  parentTranscriptionId?: string;
  isOpen: boolean;
  isLoading: boolean;
  isError: boolean;
  isCreatingEntry: boolean;
  isUpdatingEntry: boolean;
  isDeletingEntry: boolean;
  width: number;
  onWidthChange: (width: number) => void;
  onCreateEntry: (annotationId: string, content: string) => Promise<void>;
  onUpdateEntry: (annotationId: string, entryId: string, content: string) => Promise<void>;
  onDeleteEntry: (annotationId: string, entryId: string) => Promise<void>;
  onSeekRequest: (seconds: number) => void;
  onOpenChange: (isOpen: boolean) => void;
};

type ResizeCorner = "top-left" | "top-right" | "bottom-left" | "bottom-right";
type PanePosition = { x: number; y: number };

const assistantHeightStorageKey = "scriberr.audioDetail.assistantHeight";
const assistantPositionStorageKey = "scriberr.audioDetail.assistantPosition";

function clampPaneHeight(height: number) {
  if (typeof window === "undefined") return Math.min(Math.max(Math.round(height), 420), 720);
  const maxHeight = Math.max(420, Math.min(760, window.innerHeight - 112));
  return Math.min(Math.max(Math.round(height), 420), maxHeight);
}

function getStoredPaneHeight() {
  if (typeof window === "undefined") return 680;
  const storedHeight = Number(window.localStorage.getItem(assistantHeightStorageKey));
  if (Number.isFinite(storedHeight) && storedHeight > 0) return clampPaneHeight(storedHeight);
  return clampPaneHeight(680);
}

function getStoredPanePosition(): PanePosition {
  if (typeof window === "undefined") return { x: 0, y: 0 };
  try {
    const rawPosition = window.localStorage.getItem(assistantPositionStorageKey);
    if (!rawPosition) return { x: 0, y: 0 };
    const parsed = JSON.parse(rawPosition) as Partial<PanePosition>;
    const x = Number(parsed.x);
    const y = Number(parsed.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return { x: 0, y: 0 };
    return { x, y };
  } catch {
    return { x: 0, y: 0 };
  }
}

export function TranscriptNotesSidebar({
  notes,
  parentTranscriptionId,
  isOpen,
  isLoading,
  isError,
  isCreatingEntry,
  isUpdatingEntry,
  isDeletingEntry,
  width,
  onWidthChange,
  onCreateEntry,
  onUpdateEntry,
  onDeleteEntry,
  onSeekRequest,
  onOpenChange,
}: TranscriptNotesSidebarProps) {
  const [activeReplyNoteId, setActiveReplyNoteId] = useState<string | null>(null);
  const [activePanel, setActivePanel] = useState<"chat" | "notes">("chat");
  const [paneHeight, setPaneHeight] = useState(getStoredPaneHeight);
  const [resizeState, setResizeState] = useState<{
    corner: ResizeCorner;
    startX: number;
    startY: number;
    startWidth: number;
    startHeight: number;
    originY: number;
  } | null>(null);
  const [dragState, setDragState] = useState<{ startX: number; startY: number; originX: number; originY: number } | null>(null);
  const [position, setPosition] = useState<PanePosition>(getStoredPanePosition);

  useEffect(() => {
    window.localStorage.setItem(assistantHeightStorageKey, String(paneHeight));
  }, [paneHeight]);

  useEffect(() => {
    window.localStorage.setItem(assistantPositionStorageKey, JSON.stringify(position));
  }, [position]);

  useEffect(() => {
    if (!resizeState) return;

    const handlePointerMove = (event: globalThis.PointerEvent) => {
      const deltaX = event.clientX - resizeState.startX;
      const deltaY = event.clientY - resizeState.startY;
      const isLeft = resizeState.corner.includes("left");
      const isTop = resizeState.corner.includes("top");
      onWidthChange(resizeState.startWidth + (isLeft ? -deltaX : deltaX));
      setPaneHeight(clampPaneHeight(resizeState.startHeight + (isTop ? -deltaY : deltaY)));
      if (isTop) {
        setPosition((current) => ({ ...current, y: resizeState.originY + deltaY }));
      }
    };
    const handlePointerUp = () => {
      setResizeState(null);
    };

    document.body.dataset.notesSidebarResizing = resizeState.corner;
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp, { once: true });
    window.addEventListener("pointercancel", handlePointerUp, { once: true });

    return () => {
      delete document.body.dataset.notesSidebarResizing;
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [resizeState, onWidthChange]);

  useEffect(() => {
    if (!dragState) return;

    const handlePointerMove = (event: globalThis.PointerEvent) => {
      const nextX = dragState.originX + event.clientX - dragState.startX;
      const nextY = dragState.originY + event.clientY - dragState.startY;
      const viewportInset = 16;
      const baseRight = 40;
      const baseTop = 88;
      const estimatedWidth = Math.min(width, window.innerWidth - viewportInset * 2);
      const estimatedHeight = Math.min(paneHeight, window.innerHeight - viewportInset * 2);
      setPosition({
        x: Math.min(Math.max(nextX, -window.innerWidth + estimatedWidth + viewportInset + baseRight), baseRight - viewportInset),
        y: Math.min(Math.max(nextY, viewportInset - baseTop), window.innerHeight - estimatedHeight - viewportInset - baseTop),
      });
    };
    const handlePointerUp = () => {
      setDragState(null);
    };

    document.body.dataset.notesPaneDragging = "true";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp, { once: true });
    window.addEventListener("pointercancel", handlePointerUp, { once: true });

    return () => {
      delete document.body.dataset.notesPaneDragging;
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [dragState, paneHeight, width]);

  const handleResizePointerDown = (event: PointerEvent<HTMLDivElement>, corner: ResizeCorner) => {
    if (!isOpen) return;
    event.preventDefault();
    setResizeState({
      corner,
      startX: event.clientX,
      startY: event.clientY,
      startWidth: width,
      startHeight: paneHeight,
      originY: position.y,
    });
  };

  const handleResizeKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!isOpen) return;
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const step = event.shiftKey ? 48 : 16;
    onWidthChange(width + (event.key === "ArrowRight" ? step : -step));
  };

  const handleDragPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!isOpen) return;
    const target = event.target as HTMLElement;
    if (target.closest("button, a, input, textarea, select, [role='button']")) return;
    event.preventDefault();
    setDragState({
      startX: event.clientX,
      startY: event.clientY,
      originX: position.x,
      originY: position.y,
    });
  };

  if (!isOpen) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            className="scr-transcript-notes-toggle"
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Open chat and notes"
            aria-expanded={false}
            onClick={() => onOpenChange(true)}
          >
            <MessageSquareText size={18} aria-hidden="true" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Chat and notes</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <aside
      className="scr-transcript-notes-sidebar"
      data-open="true"
      aria-label="Transcript chat and notes"
      style={{
        "--scr-floating-pane-width": `${width}px`,
        "--scr-floating-pane-height": `${paneHeight}px`,
        transform: `translate3d(${position.x}px, ${position.y}px, 0)`,
      } as CSSProperties}
    >
      {(["top-left", "top-right", "bottom-left", "bottom-right"] as const).map((corner) => (
        <div
          className="scr-transcript-notes-resize-handle"
          data-corner={corner}
          key={corner}
          role="separator"
          aria-label="Resize chat and notes pane"
          aria-valuenow={Math.round(width)}
          tabIndex={corner === "bottom-right" ? 0 : -1}
          onPointerDown={(event) => handleResizePointerDown(event, corner)}
          onKeyDown={handleResizeKeyDown}
        />
      ))}
      <div className="scr-transcript-notes-dragbar" onPointerDown={handleDragPointerDown}>
        <GripHorizontal size={18} aria-hidden="true" />
        <span>Assistant</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              className="scr-transcript-notes-close"
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Close chat and notes"
              onClick={() => onOpenChange(false)}
            >
              <X size={16} aria-hidden="true" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Close</TooltipContent>
        </Tooltip>
      </div>

      <div className="scr-transcript-notes-panel">
        <nav className="scr-transcript-notes-tabs" aria-label="Detail pane">
          <button type="button" data-active={activePanel === "chat" ? "true" : undefined} onClick={() => setActivePanel("chat")}>Chat</button>
          <button type="button" data-active={activePanel === "notes" ? "true" : undefined} onClick={() => setActivePanel("notes")}>Notes</button>
        </nav>

        {activePanel === "chat" ? (
          <TranscriptChatPanel parentTranscriptionId={parentTranscriptionId} />
        ) : (
          <div className="scr-transcript-notes-list">
            {isLoading ? <p className="scr-transcript-notes-status">Loading notes.</p> : null}
            {isError ? <p className="scr-transcript-notes-status">Notes could not be loaded.</p> : null}
            {!isLoading && !isError && notes.length === 0 ? (
              <p className="scr-transcript-notes-status">No notes yet.</p>
            ) : null}
            {!isLoading && !isError ? notes.map((note) => (
              <TranscriptNoteItem
                key={note.id}
                note={note}
                isReplyActive={activeReplyNoteId === note.id}
                isCreatingEntry={isCreatingEntry && activeReplyNoteId === note.id}
                isUpdatingEntry={isUpdatingEntry}
                isDeletingEntry={isDeletingEntry}
                onActivateReply={() => setActiveReplyNoteId(note.id)}
                onCancelReply={() => setActiveReplyNoteId(null)}
                onCreateEntry={async (content) => {
                  setActiveReplyNoteId(note.id);
                  await onCreateEntry(note.id, content);
                  setActiveReplyNoteId(null);
                }}
                onUpdateEntry={onUpdateEntry}
                onDeleteEntry={onDeleteEntry}
                onSeekRequest={onSeekRequest}
              />
            )) : null}
          </div>
        )}
      </div>
    </aside>
  );
}

type TranscriptNoteItemProps = {
  note: TranscriptNoteAnnotation;
  isReplyActive: boolean;
  isCreatingEntry: boolean;
  isUpdatingEntry: boolean;
  isDeletingEntry: boolean;
  onActivateReply: () => void;
  onCancelReply: () => void;
  onCreateEntry: (content: string) => Promise<void>;
  onUpdateEntry: (annotationId: string, entryId: string, content: string) => Promise<void>;
  onDeleteEntry: (annotationId: string, entryId: string) => Promise<void>;
  onSeekRequest: (seconds: number) => void;
};

function TranscriptNoteItem({
  note,
  isReplyActive,
  isCreatingEntry,
  isUpdatingEntry,
  isDeletingEntry,
  onActivateReply,
  onCancelReply,
  onCreateEntry,
  onUpdateEntry,
  onDeleteEntry,
  onSeekRequest,
}: TranscriptNoteItemProps) {
  const timeLabel = formatAnnotationTime(note.anchor.start_ms);
  const seekSeconds = Math.max(0, note.anchor.start_ms / 1000);
  const noteCount = note.entries.length;
  const [replyContent, setReplyContent] = useState("");
  const canSubmitReply = replyContent.trim().length > 0 && !isCreatingEntry;

  const handleReplySubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = replyContent.trim();
    if (!content || isCreatingEntry) return;
    await onCreateEntry(content);
    setReplyContent("");
  };

  const handleReplyKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setReplyContent("");
      onCancelReply();
      event.currentTarget.blur();
      return;
    }
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <article className="scr-transcript-note-item">
      <h3>{note.quote}</h3>
      <div className="scr-transcript-note-meta">
        <button
          className="scr-transcript-note-time"
          type="button"
          aria-label={`Seek to ${timeLabel}`}
          onClick={() => onSeekRequest(seekSeconds)}
        >
          <Mic2 size={16} aria-hidden="true" />
          {timeLabel}
        </button>
        <span className="scr-transcript-note-count" aria-label={`${noteCount} ${noteCount === 1 ? "note" : "notes"}`}>
          <MessageCircle size={16} aria-hidden="true" />
          {noteCount}
        </span>
      </div>
      {note.entries.map((entry) => (
        <TranscriptNoteEntryBubble
          key={entry.id}
          annotationId={note.id}
          entry={entry}
          isUpdating={isUpdatingEntry}
          isDeleting={isDeletingEntry}
          onUpdate={onUpdateEntry}
          onDelete={onDeleteEntry}
        />
      ))}
      <form className="scr-transcript-note-reply" onSubmit={handleReplySubmit}>
        <input
          className="scr-transcript-note-reply-input"
          value={replyContent}
          aria-label={`Reply to note at ${timeLabel}`}
          placeholder="Reply..."
          disabled={isCreatingEntry}
          onChange={(event) => setReplyContent(event.currentTarget.value)}
          onFocus={onActivateReply}
          onKeyDown={handleReplyKeyDown}
        />
        <Button
          className="scr-transcript-note-reply-submit"
          type="submit"
          variant="ghost"
          size="icon"
          aria-label="Send reply"
          disabled={!canSubmitReply || !isReplyActive}
        >
          <Send size={16} aria-hidden="true" />
        </Button>
      </form>
    </article>
  );
}

type TranscriptNoteEntryBubbleProps = {
  annotationId: string;
  entry: TranscriptNoteAnnotation["entries"][number];
  isUpdating: boolean;
  isDeleting: boolean;
  onUpdate: (annotationId: string, entryId: string, content: string) => Promise<void>;
  onDelete: (annotationId: string, entryId: string) => Promise<void>;
};

function TranscriptNoteEntryBubble({ annotationId, entry, isUpdating, isDeleting, onUpdate, onDelete }: TranscriptNoteEntryBubbleProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draftContent, setDraftContent] = useState(entry.content);
  const trimmedDraft = draftContent.trim();
  const canSave = trimmedDraft.length > 0 && trimmedDraft !== entry.content.trim() && !isUpdating;

  useEffect(() => {
    if (!isEditing) setDraftContent(entry.content);
  }, [entry.content, isEditing]);

  const handleSave = async () => {
    if (!canSave) {
      setDraftContent(entry.content);
      setIsEditing(false);
      return;
    }
    await onUpdate(annotationId, entry.id, trimmedDraft);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setDraftContent(entry.content);
    setIsEditing(false);
  };

  const handleDelete = async () => {
    if (isDeleting) return;
    await onDelete(annotationId, entry.id);
  };

  const handleEditKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      handleCancel();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.blur();
    }
  };

  if (isEditing) {
    return (
      <div className="scr-transcript-note-entry" data-editing="true">
        <textarea
          className="scr-transcript-note-edit-input"
          value={draftContent}
          aria-label="Edit note"
          disabled={isUpdating}
          autoFocus
          rows={1}
          onBlur={() => void handleSave()}
          onChange={(event) => setDraftContent(event.currentTarget.value)}
          onKeyDown={handleEditKeyDown}
        />
      </div>
    );
  }

  return (
    <div className="scr-transcript-note-entry">
      <button className="scr-transcript-note-entry-content" type="button" onClick={() => setIsEditing(true)}>
        {entry.content}
      </button>
      <div className="scr-transcript-note-entry-actions">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              className="scr-transcript-note-entry-action scr-transcript-note-entry-delete"
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Delete note"
              disabled={isDeleting}
              onClick={handleDelete}
            >
              <Trash2 size={14} aria-hidden="true" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Delete</TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}

function formatAnnotationTime(milliseconds: number) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
