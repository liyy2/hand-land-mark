# Clinical Video Annotation Tool – User Guide

## Getting Started
- Launch the app and use the front page to upload a video file or enter a path. Wait for processing to finish, then click “Start Annotation”.
- The version (git commit) is shown on the upload page for traceability.

## Layout Overview
- **Top toolbar:** New Video, Add Annotation, Duplicate, Delete, Save, Open, Shortcuts.
- **Video preview:** Playback controls and speed controls under the video.
- **Properties panel (right):** Edit the selected annotation’s timing, task, severity, track, and notes. Quick buttons set start/end to the current playhead.
- **Timeline:** Ruler at top, multiple tracks below. Red playhead shows current time. Annotations are color-coded by task. Selected annotations glow bright yellow.

## Creating & Editing Annotations
- Click **Add Annotation** (or press `M`) to create at the current playhead. Choose task, duration, severity, and track.
- Drag annotations to reposition; drag edges to resize. Drag vertically to move between tracks.
- Context menu (right-click an annotation): Cut/Copy/Paste, Delete, Split at Playhead, Set Start/End @ Current, Move to Previous/Next Track.
- Properties panel timing buttons: “Set Start @ Current” and “Set End @ Current”.

## Track Management
- Use **Add Track** / **Remove Track** in the timeline header. Tracks auto-create when you move an annotation to a new track.
- Shortcuts: `Cmd/Ctrl + ↓` moves selection to the next track (creates one if needed); `Cmd/Ctrl + ↑` moves to the previous track.

## Saving & Loading
- **Save (Cmd/Ctrl + S):** Downloads a JSON with annotations plus metadata (`videoName`, `videoDurationSeconds`).
- **Open:** Click “Open” to pick Append or Start Fresh, then choose a JSON. If the saved duration differs from the current video, you’ll get a warning.
- Loading always keeps track indices; the timeline adds tracks if necessary.

## Playback & Navigation
- Play/Pause: `Space`
- Split at playhead: `S`
- Delete selected: `Delete`/`Backspace`
- Duplicate: `Cmd/Ctrl + D`
- Copy/Paste: `Cmd/Ctrl + C / V`
- Set Start/End to playhead: `[` / `]`
- Add Annotation: `M`
- Drag the red playhead to scrub; zoom with +/− buttons; Fit to window to auto-scale.

## Tips
- Selection glow may extend outside the track; overflow is allowed so you can see it even at time 0.
- If dragging feels slow, keep the pointer over the annotation body (grab cursor). Handles are wider for easier resizing.
- Color coding is deterministic per task, so the same task always has the same hue.
