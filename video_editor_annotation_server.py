#!/usr/bin/env python
"""
Professional Video Editor-Style Annotation Server
Timeline-based interface similar to Final Cut Pro / Premiere Pro
"""

from flask import Flask, render_template_string, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile

app = Flask(__name__)
CORS(app)

# Store annotations and video info
annotations = []
current_video_path = None
video_info = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Professional Video Annotation Editor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #e0e0e0;
            overflow-x: hidden;
        }
        
        /* Top Toolbar */
        .toolbar {
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            padding: 8px 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .toolbar-title {
            font-size: 14px;
            font-weight: 600;
            color: #cccccc;
            margin-right: auto;
        }
        
        .toolbar-btn {
            background: #3c3c3c;
            border: 1px solid #464647;
            color: #cccccc;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        
        .toolbar-btn:hover {
            background: #464647;
            border-color: #5a5a5a;
        }
        
        .toolbar-btn.active {
            background: #007acc;
            border-color: #007acc;
            color: white;
        }
        
        /* Main Layout */
        .main-container {
            display: flex;
            height: calc(100vh - 50px);
        }
        
        /* Video Preview Section */
        .video-section {
            flex: 1;
            background: #252526;
            display: flex;
            flex-direction: column;
        }
        
        .video-container {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1e1e1e;
        }
        
        video {
            max-width: 100%;
            max-height: 100%;
            background: #000;
        }
        
        /* Video Controls */
        .video-controls {
            background: #2d2d30;
            border-top: 1px solid #3e3e42;
            padding: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .play-btn {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #007acc;
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .play-btn:hover {
            background: #1a86d3;
        }
        
        .time-display {
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            color: #cccccc;
            min-width: 150px;
            text-align: center;
            background: #1e1e1e;
            padding: 5px 10px;
            border-radius: 4px;
        }
        
        .speed-control {
            margin-left: auto;
        }
        
        .speed-btn {
            background: #3c3c3c;
            border: 1px solid #464647;
            color: #cccccc;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            margin: 0 2px;
        }
        
        /* Timeline Section */
        .timeline-section {
            background: #252526;
            border-top: 1px solid #3e3e42;
            height: 300px;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        
        .timeline-header {
            background: #2d2d30;
            padding: 8px 15px;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .zoom-controls {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .zoom-btn {
            background: #3c3c3c;
            border: 1px solid #464647;
            color: #cccccc;
            width: 24px;
            height: 24px;
            border-radius: 3px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .timeline-workspace {
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #1e1e1e;
        }
        
        .timeline-scrollable {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            overflow-x: auto;
            overflow-y: hidden;
        }
        
        .timeline-content {
            position: relative;
            height: 100%;
            min-width: 100%;
        }
        
        /* Timeline Ruler */
        .timeline-ruler {
            height: 30px;
            background: #2d2d30;
            border-bottom: 1px solid #3e3e42;
            position: relative;
            user-select: none;
        }
        
        /* Video duration indicator */
        .video-duration-bg {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: linear-gradient(90deg, #2d2d30 0%, #353538 100%);
            border-right: 3px solid #007acc;
            z-index: 1;
        }
        
        .video-end-marker {
            position: absolute;
            top: 0;
            width: 3px;
            height: 100%;
            background: #007acc;
            z-index: 5;
        }
        
        .video-end-label {
            position: absolute;
            top: -20px;
            padding: 2px 6px;
            background: #007acc;
            color: white;
            font-size: 10px;
            border-radius: 3px;
            transform: translateX(-50%);
            white-space: nowrap;
        }
        
        .ruler-tick {
            position: absolute;
            top: 0;
            width: 1px;
            height: 100%;
            background: #464647;
        }
        
        .ruler-label {
            position: absolute;
            top: 5px;
            font-size: 10px;
            color: #969696;
            transform: translateX(-50%);
        }
        
        /* Timeline Tracks */
        .timeline-tracks {
            position: relative;
            padding: 10px 0;
        }
        
        .timeline-track {
            height: 40px;
            margin: 4px 0;
            position: relative;
            background: #1e1e1e;
            border: 1px solid #3e3e42;
            border-radius: 2px;
            overflow: hidden;
        }
        
        .track-duration-bg {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: rgba(0, 122, 204, 0.05);
            border-right: 1px solid rgba(0, 122, 204, 0.3);
            z-index: 1;
        }
        
        .track-label {
            position: absolute;
            left: -120px;
            top: 50%;
            transform: translateY(-50%);
            width: 110px;
            text-align: right;
            font-size: 12px;
            color: #969696;
            padding-right: 10px;
        }
        
        /* Timeline Segments */
        .timeline-segment {
            position: absolute;
            height: 100%;
            background: #007acc;
            border: 2px solid #1a86d3;
            border-radius: 4px;
            cursor: move;
            user-select: none;
            display: flex;
            align-items: center;
            padding: 0 8px;
            overflow: hidden;
            transition: box-shadow 0.2s;
            z-index: 5; /* Base z-index for segments */
        }
        
        .timeline-segment:hover {
            box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.3);
            z-index: 10;
        }
        
        .timeline-segment.selected {
            border-color: #ffc107;
            box-shadow: 0 0 0 3px rgba(255, 193, 7, 0.3);
            z-index: 20;
        }
        
        .timeline-segment.dragging {
            opacity: 0.8;
            z-index: 30;
        }
        
        .segment-label {
            font-size: 11px;
            color: white;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            pointer-events: none;
        }
        
        .segment-resize-handle {
            position: absolute;
            top: 0;
            width: 8px;
            height: 100%;
            cursor: ew-resize;
            background: transparent;
        }
        
        .segment-resize-handle:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .segment-resize-handle.left {
            left: 0;
        }
        
        .segment-resize-handle.right {
            right: 0;
        }
        
        /* Playhead */
        .playhead {
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            background: #ff3333;
            cursor: ew-resize;
            z-index: 50;
            will-change: left;
        }
        
        .playhead.dragging {
            /* Instant response when dragging */
            transition: none !important;
        }
        
        .playhead::before {
            content: '';
            position: absolute;
            top: -8px;
            left: -6px; /* Shifted slightly right for better alignment */
            width: 14px;
            height: 14px;
            background: #ff3333;
            transform: rotate(45deg);
            cursor: ew-resize;
            pointer-events: none; /* Don't block clicks on segments */
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .playhead:hover {
            width: 3px;
            box-shadow: 0 0 8px rgba(255, 51, 51, 0.5);
        }
        
        .playhead:hover::before {
            transform: rotate(45deg) scale(1.2);
        }
        
        /* Properties Panel */
        .properties-panel {
            width: 300px;
            background: #252526;
            border-left: 1px solid #3e3e42;
            display: flex;
            flex-direction: column;
        }
        
        .panel-header {
            background: #2d2d30;
            padding: 10px 15px;
            border-bottom: 1px solid #3e3e42;
            font-size: 13px;
            font-weight: 600;
        }
        
        .panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }
        
        .property-group {
            margin-bottom: 20px;
        }
        
        .property-label {
            font-size: 11px;
            color: #969696;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .property-input {
            width: 100%;
            background: #3c3c3c;
            border: 1px solid #464647;
            color: #cccccc;
            padding: 6px 8px;
            border-radius: 3px;
            font-size: 12px;
        }
        
        .property-input:focus {
            outline: none;
            border-color: #007acc;
        }
        
        /* Task Type Colors - Comprehensive list */
        .segment-finger-tapping { background: linear-gradient(135deg, #4c9aff, #2684ff); }
        .segment-hand-opening-closing { background: linear-gradient(135deg, #69f0ae, #00e676); }
        .segment-pronation-supination { background: linear-gradient(135deg, #ffab40, #ff9100); }
        .segment-rest-tremor { background: linear-gradient(135deg, #ff5252, #ff1744); }
        .segment-postural-tremor { background: linear-gradient(135deg, #ff6b6b, #ff3838); }
        .segment-kinetic-tremor { background: linear-gradient(135deg, #ff8787, #ff5555); }
        .segment-gait { background: linear-gradient(135deg, #b388ff, #7c4dff); }
        .segment-facial-expression { background: linear-gradient(135deg, #ff80ab, #ff4081); }
        .segment-toe-tapping { background: linear-gradient(135deg, #81c784, #4caf50); }
        .segment-leg-agility { background: linear-gradient(135deg, #4dd0e1, #00acc1); }
        .segment-speech { background: linear-gradient(135deg, #ffb74d, #ff9800); }
        .segment-writing { background: linear-gradient(135deg, #9575cd, #673ab7); }
        .segment-other { background: linear-gradient(135deg, #90a4ae, #607d8b); }
        
        /* Context Menu */
        .context-menu {
            position: fixed;
            background: #2d2d30;
            border: 1px solid #464647;
            border-radius: 4px;
            padding: 4px 0;
            min-width: 150px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            z-index: 1000;
            display: none;
        }
        
        .context-menu-item {
            padding: 6px 20px;
            font-size: 12px;
            color: #cccccc;
            cursor: pointer;
        }
        
        .context-menu-item:hover {
            background: #094771;
            color: white;
        }
        
        .context-menu-separator {
            height: 1px;
            background: #464647;
            margin: 4px 0;
        }
        
        /* Annotation Dialog - positioned on right */
        .annotation-dialog {
            display: none;
            position: fixed;
            top: 80px;
            right: 20px;
            background: #2d2d30;
            border: 1px solid #464647;
            border-radius: 8px;
            padding: 20px;
            width: 400px;
            max-width: 90vw;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        
        .annotation-dialog h3 {
            margin-bottom: 15px;
            color: #cccccc;
        }
        
        /* Keyboard Shortcuts Modal */
        .shortcuts-modal {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #2d2d30;
            border: 1px solid #464647;
            border-radius: 8px;
            padding: 20px;
            max-width: 400px;
            z-index: 1000;
        }
        
        .shortcuts-modal h3 {
            margin-bottom: 15px;
            color: #cccccc;
        }
        
        .shortcut-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 12px;
        }
        
        .shortcut-key {
            background: #1e1e1e;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            color: #ffc107;
        }
        
        /* Loading Spinner */
        .loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #007acc;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <!-- Top Toolbar -->
    <div class="toolbar">
        <div class="toolbar-title">📹 Clinical Video Annotation Tool</div>
        <button class="toolbar-btn" onclick="showNewAnnotationDialog()" style="background: #007acc; color: white;">➕ Add Annotation</button>
        <button class="toolbar-btn" onclick="duplicateSelected()">📋 Duplicate</button>
        <button class="toolbar-btn" onclick="deleteSelected()">🗑️ Delete</button>
        <div style="width: 1px; height: 20px; background: #464647;"></div>
        <button class="toolbar-btn" onclick="saveProject()">💾 Save</button>
        <button class="toolbar-btn" onclick="loadProject()">📁 Open</button>
        <button class="toolbar-btn" onclick="exportAnnotations()">📤 Export</button>
        <div style="width: 1px; height: 20px; background: #464647;"></div>
        <button class="toolbar-btn" onclick="showShortcuts()">⌨️ Shortcuts</button>
    </div>
    
    <div class="main-container">
        <!-- Video Preview -->
        <div class="video-section">
            <div class="video-container">
                <video id="videoPlayer">
                    <source src="{{ video_url }}" type="video/mp4">
                </video>
            </div>
            
            <div class="video-controls">
                <button class="play-btn" id="playBtn" onclick="togglePlayPause()">▶</button>
                <div class="time-display" id="timeDisplay">00:00:00 / 00:00:00</div>
                <div class="speed-control">
                    <button class="speed-btn" onclick="setSpeed(0.25)">0.25x</button>
                    <button class="speed-btn" onclick="setSpeed(0.5)">0.5x</button>
                    <button class="speed-btn" onclick="setSpeed(1)" style="background: #007acc;">1x</button>
                    <button class="speed-btn" onclick="setSpeed(1.5)">1.5x</button>
                    <button class="speed-btn" onclick="setSpeed(2)">2x</button>
                </div>
            </div>
        </div>
        
        <!-- Properties Panel -->
        <div class="properties-panel">
            <div class="panel-header">Properties Inspector</div>
            <div class="panel-content" id="propertiesContent">
                <div class="property-group">
                    <div class="property-label">No Selection</div>
                    <p style="font-size: 12px; color: #969696;">Select a segment to edit its properties</p>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Timeline -->
    <div class="timeline-section">
        <div class="timeline-header">
            <span style="font-size: 12px; color: #969696;">Timeline</span>
            <button class="toolbar-btn" onclick="toggleSecondTrack()" id="trackToggleBtn">Add Track 2</button>
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomOut()">−</button>
                <span style="font-size: 11px; color: #969696; padding: 0 8px;">Zoom</span>
                <button class="zoom-btn" onclick="zoomIn()">+</button>
            </div>
            <button class="toolbar-btn" onclick="fitToWindow()">Fit</button>
        </div>
        
        <div class="timeline-workspace">
            <div class="timeline-scrollable" id="timelineScrollable">
                <div class="timeline-content" id="timelineContent">
                    <!-- Ruler -->
                    <div class="timeline-ruler" id="timelineRuler" style="margin-left: 130px;"></div>
                    
                    <!-- Timeline Tracks -->
                    <div class="timeline-tracks" id="timelineTracks" style="margin-left: 130px;">
                        <!-- Track 1 -->
                        <div class="timeline-track" data-track="0" style="height: 60px;">
                            <div class="track-label">Track 1</div>
                        </div>
                        <!-- Track 2 (optional) -->
                        <div class="timeline-track" data-track="1" style="height: 60px; display: none;">
                            <div class="track-label">Track 2</div>
                        </div>
                    </div>
                    
                    <!-- Playhead -->
                    <div class="playhead" id="playhead" style="left: 130px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Context Menu -->
    <div class="context-menu" id="contextMenu">
        <div class="context-menu-item" onclick="cutSegment()">Cut</div>
        <div class="context-menu-item" onclick="copySegment()">Copy</div>
        <div class="context-menu-item" onclick="pasteSegment()">Paste</div>
        <div class="context-menu-separator"></div>
        <div class="context-menu-item" onclick="deleteSelected()">Delete</div>
        <div class="context-menu-separator"></div>
        <div class="context-menu-item" onclick="splitAtPlayhead()">Split at Playhead</div>
    </div>
    
    <!-- Shortcuts Modal -->
    <div class="shortcuts-modal" id="shortcutsModal">
        <h3>⌨️ Keyboard Shortcuts</h3>
        <div class="shortcut-item">
            <span>Play/Pause</span>
            <span class="shortcut-key">Space</span>
        </div>
        <div class="shortcut-item">
            <span>Add Marker</span>
            <span class="shortcut-key">M</span>
        </div>
        <div class="shortcut-item">
            <span>Split at Playhead</span>
            <span class="shortcut-key">S</span>
        </div>
        <div class="shortcut-item">
            <span>Delete Selected</span>
            <span class="shortcut-key">Delete</span>
        </div>
        <div class="shortcut-item">
            <span>Duplicate</span>
            <span class="shortcut-key">Cmd/Ctrl + D</span>
        </div>
        <div class="shortcut-item">
            <span>Save Project</span>
            <span class="shortcut-key">Cmd/Ctrl + S</span>
        </div>
        <button class="toolbar-btn" onclick="hideShortcuts()" style="margin-top: 15px; width: 100%;">Close</button>
    </div>
    
    <!-- New Annotation Dialog -->
    <div class="annotation-dialog" id="newAnnotationDialog">
        <h3>➕ Add New Annotation</h3>
        <div style="margin: 20px 0;">
            <label style="display: block; color: #969696; margin-bottom: 5px;">Task Type</label>
            <select id="newTaskType" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
                <option value="Finger Tapping">Finger Tapping</option>
                <option value="Hand Opening/Closing">Hand Opening/Closing</option>
                <option value="Pronation-Supination">Pronation-Supination</option>
                <option value="Rest Tremor">Rest Tremor</option>
                <option value="Postural Tremor">Postural Tremor</option>
                <option value="Kinetic Tremor">Kinetic Tremor</option>
                <option value="Gait">Gait</option>
                <option value="Facial Expression">Facial Expression</option>
                <option value="Toe Tapping">Toe Tapping</option>
                <option value="Leg Agility">Leg Agility</option>
                <option value="Speech">Speech</option>
                <option value="Writing">Writing</option>
            </select>
            
            <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Side</label>
            <select id="newSide" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
                <option value="bilateral">Bilateral</option>
                <option value="left">Left</option>
                <option value="right">Right</option>
                <option value="n/a">N/A</option>
            </select>
            
            <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Duration (seconds)</label>
            <input type="number" id="newDuration" value="5" min="0.5" max="30" step="0.5" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
            
            <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Severity (UPDRS 0-4)</label>
            <input type="number" id="newSeverity" value="0" min="0" max="4" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
            
            <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Track</label>
            <select id="newTrack" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
                <option value="0">Track 1</option>
                <option value="1" id="track2Option" style="display: none;">Track 2</option>
            </select>
        </div>
        <div style="display: flex; gap: 10px;">
            <button class="toolbar-btn" onclick="createNewAnnotation()" style="background: #007acc; color: white; flex: 1;">Create at Current Time</button>
            <button class="toolbar-btn" onclick="hideNewAnnotationDialog()" style="flex: 1;">Cancel</button>
        </div>
    </div>
    
    <script>
        // Global variables
        let video = document.getElementById('videoPlayer');
        let selectedSegment = null;
        let isDragging = false;
        let isResizing = false;
        let isPlayheadDragging = false; // Track playhead dragging state globally
        let justFinishedDraggingPlayhead = false; // Track if we just finished dragging
        let dragStartX = 0;
        let segmentStartPos = 0;
        let segmentStartWidth = 0;
        let timelineZoom = 10; // pixels per second
        let videoDuration = 0;
        let annotations = [];
        let clipboard = null;
        let hasSecondTrack = false; // Track if second track is visible
        
        // Store last used settings for convenience (but always allow changing)
        let lastUsedSettings = {
            task: 'Finger Tapping',
            side: 'bilateral',
            duration: 5,
            severity: 0
        };
        
        // Initialize video
        video.addEventListener('loadedmetadata', () => {
            videoDuration = video.duration;
            
            // Set initial zoom to fit video in view with some padding
            const scrollable = document.getElementById('timelineScrollable');
            const availableWidth = scrollable.clientWidth - 130; // Account for labels
            // Make video take up 80% of available width for better visibility
            timelineZoom = (availableWidth * 0.8) / videoDuration;
            // But ensure minimum zoom for very short videos
            timelineZoom = Math.max(10, timelineZoom);
            
            updateTimeline();
            updateTimeDisplay();
            
            // Initialize playhead at correct position (0:00 = 130px due to label margin)
            const playhead = document.getElementById('playhead');
            playhead.style.left = '130px';
            
            // Make playhead draggable
            initPlayheadDragging();
        });
        
        video.addEventListener('timeupdate', () => {
            updatePlayhead();
            updateTimeDisplay();
        });
        
        // Playhead dragging
        function initPlayheadDragging() {
            const playhead = document.getElementById('playhead');
            let rafId = null;
            
            // Prevent playhead clicks from bubbling to timeline
            playhead.addEventListener('click', (e) => {
                e.stopPropagation();
                e.stopImmediatePropagation();
            });
            
            playhead.addEventListener('mousedown', (e) => {
                isPlayheadDragging = true; // Use global flag
                e.stopPropagation();
                e.stopImmediatePropagation();
                e.preventDefault();
                
                // Add dragging class for instant response
                playhead.classList.add('dragging');
                
                // Store the initial mouse position
                let currentMouseX = e.clientX;
                
                const handleDrag = (e) => {
                    if (!isPlayheadDragging) return;
                    
                    currentMouseX = e.clientX;
                    
                    // Use requestAnimationFrame for smooth updates
                    if (rafId) cancelAnimationFrame(rafId);
                    rafId = requestAnimationFrame(() => {
                        const timeline = document.getElementById('timelineContent');
                        const rect = timeline.getBoundingClientRect();
                        const scrollLeft = document.getElementById('timelineScrollable').scrollLeft;
                        const x = currentMouseX - rect.left + scrollLeft - 130; // Account for track labels and scroll
                        const time = Math.max(0, Math.min(videoDuration, x / timelineZoom));
                        
                        // Update video time smoothly
                        video.currentTime = time;
                        
                        // Update playhead position immediately for visual feedback
                        playhead.style.left = (time * timelineZoom + 130) + 'px';
                    });
                };
                
                const stopDrag = () => {
                    playhead.classList.remove('dragging');
                    if (rafId) {
                        cancelAnimationFrame(rafId);
                        rafId = null;
                    }
                    document.removeEventListener('mousemove', handleDrag);
                    document.removeEventListener('mouseup', stopDrag);
                    
                    // Mark that we just finished dragging
                    isPlayheadDragging = false;
                    justFinishedDraggingPlayhead = true;
                    
                    // Reset the flag after a short delay to ignore the click event
                    setTimeout(() => {
                        justFinishedDraggingPlayhead = false;
                    }, 100);
                };
                
                document.addEventListener('mousemove', handleDrag);
                document.addEventListener('mouseup', stopDrag);
            });
            
            // Handle clicking on the timeline for selection/deselection only
            const timelineContent = document.getElementById('timelineContent');
            timelineContent.addEventListener('click', (e) => {
                // Don't process clicks if we were just dragging the playhead
                if (isPlayheadDragging || justFinishedDraggingPlayhead) {
                    return;
                }
                
                // Don't do anything if clicking on a segment or its children
                if (e.target.classList.contains('timeline-segment') || 
                    e.target.classList.contains('segment-label') ||
                    e.target.classList.contains('segment-resize-handle') ||
                    e.target.id === 'playhead' ||
                    e.target.classList.contains('playhead')) {
                    return;
                }
                
                // Only deselect if clicking on track background or ruler
                if (e.target.classList.contains('timeline-track') || 
                    e.target.classList.contains('track-duration-bg') ||
                    e.target.classList.contains('timeline-ruler') ||
                    e.target.classList.contains('ruler-tick') ||
                    e.target.classList.contains('ruler-label')) {
                    
                    deselectAll();
                    // NO LONGER reposition the playhead on click - only deselect
                }
            });
        }
        
        // Playback controls
        function togglePlayPause() {
            if (video.paused) {
                video.play();
                document.getElementById('playBtn').innerHTML = '⏸';
            } else {
                video.pause();
                document.getElementById('playBtn').innerHTML = '▶';
            }
        }
        
        function setSpeed(speed) {
            video.playbackRate = speed;
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.style.background = '#3c3c3c';
            });
            event.target.style.background = '#007acc';
        }
        
        function updateTimeDisplay() {
            const current = formatTime(video.currentTime);
            const total = formatTime(videoDuration);
            document.getElementById('timeDisplay').textContent = `${current} / ${total}`;
        }
        
        function formatTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
        
        // Timeline functions
        function updateTimeline() {
            const ruler = document.getElementById('timelineRuler');
            const width = Math.max(videoDuration * timelineZoom, 800); // Minimum width for visibility
            
            document.getElementById('timelineContent').style.width = width + 'px';
            
            // Clear ruler
            ruler.innerHTML = '';
            
            // Add video duration background indicator
            const durationBg = document.createElement('div');
            durationBg.className = 'video-duration-bg';
            durationBg.style.width = (videoDuration * timelineZoom) + 'px';
            ruler.appendChild(durationBg);
            
            // Add end marker with label
            const endMarker = document.createElement('div');
            endMarker.className = 'video-end-marker';
            endMarker.style.left = (videoDuration * timelineZoom) + 'px';
            ruler.appendChild(endMarker);
            
            const endLabel = document.createElement('div');
            endLabel.className = 'video-end-label';
            endLabel.style.left = (videoDuration * timelineZoom) + 'px';
            endLabel.textContent = `End: ${formatTime(videoDuration)}`;
            ruler.appendChild(endLabel);
            
            // Add ruler ticks
            const tickInterval = Math.max(1, Math.floor(50 / timelineZoom)); // Adjust tick spacing
            
            for (let i = 0; i <= videoDuration; i += tickInterval) {
                const tick = document.createElement('div');
                tick.className = 'ruler-tick';
                tick.style.left = (i * timelineZoom) + 'px';
                tick.style.zIndex = '2'; // Above background
                
                if (i % (tickInterval * 2) === 0 || i === 0) {
                    const label = document.createElement('div');
                    label.className = 'ruler-label';
                    label.style.left = (i * timelineZoom) + 'px';
                    label.style.zIndex = '3';
                    label.textContent = formatTime(i);
                    ruler.appendChild(label);
                }
                
                ruler.appendChild(tick);
            }
            
            // Update track backgrounds
            document.querySelectorAll('.timeline-track').forEach(track => {
                // Remove existing background
                const existingBg = track.querySelector('.track-duration-bg');
                if (existingBg) existingBg.remove();
                
                // Add new background
                const trackBg = document.createElement('div');
                trackBg.className = 'track-duration-bg';
                trackBg.style.width = (videoDuration * timelineZoom) + 'px';
                track.insertBefore(trackBg, track.firstChild);
            });
            
            // Update segments
            renderSegments();
        }
        
        function updatePlayhead() {
            const playhead = document.getElementById('playhead');
            const position = video.currentTime * timelineZoom + 130;
            playhead.style.left = position + 'px';
        }
        
        function zoomIn() {
            timelineZoom = Math.min(100, timelineZoom * 1.5);
            updateTimeline();
        }
        
        function zoomOut() {
            timelineZoom = Math.max(2, timelineZoom / 1.5);
            updateTimeline();
        }
        
        function fitToWindow() {
            const scrollable = document.getElementById('timelineScrollable');
            const availableWidth = scrollable.clientWidth - 130;
            timelineZoom = availableWidth / videoDuration;
            updateTimeline();
        }
        
        function toggleSecondTrack() {
            hasSecondTrack = !hasSecondTrack;
            const track2 = document.querySelector('.timeline-track[data-track="1"]');
            const track2Option = document.getElementById('track2Option');
            const trackToggleBtn = document.getElementById('trackToggleBtn');
            
            if (hasSecondTrack) {
                track2.style.display = 'block';
                track2Option.style.display = 'block';
                trackToggleBtn.textContent = 'Remove Track 2';
            } else {
                track2.style.display = 'none';
                track2Option.style.display = 'none';
                trackToggleBtn.textContent = 'Add Track 2';
                
                // Move all track 2 annotations to track 1
                annotations.forEach(ann => {
                    if (ann.track === 1) {
                        ann.track = 0;
                    }
                });
                renderSegments();
            }
            
            updateTimeline();
        }
        
        // Segment management
        function showNewAnnotationDialog() {
            // Use last settings for convenience, but ensure form is editable
            document.getElementById('newTaskType').value = lastUsedSettings.task;
            document.getElementById('newSide').value = lastUsedSettings.side;
            document.getElementById('newDuration').value = lastUsedSettings.duration.toString();
            document.getElementById('newSeverity').value = lastUsedSettings.severity.toString();
            
            // Ensure the select elements are not disabled and have all options
            const taskSelect = document.getElementById('newTaskType');
            taskSelect.disabled = false;
            
            document.getElementById('newAnnotationDialog').style.display = 'block';
        }
        
        function hideNewAnnotationDialog() {
            document.getElementById('newAnnotationDialog').style.display = 'none';
        }
        
        function createNewAnnotation() {
            const task = document.getElementById('newTaskType').value;
            const side = document.getElementById('newSide').value;
            const duration = parseFloat(document.getElementById('newDuration').value);
            const severity = parseInt(document.getElementById('newSeverity').value);
            const track = parseInt(document.getElementById('newTrack').value);
            
            // Save settings for next time (convenience feature)
            lastUsedSettings = {
                task: task,
                side: side,
                duration: duration,
                severity: severity
            };
            
            const start = video.currentTime;
            const end = Math.min(start + duration, videoDuration);
            
            // Use selected track
            const annotation = {
                id: Date.now(),
                start: start,
                end: end,
                track: track,  // Use selected track
                task: task,
                side: side,
                severity: severity,
                notes: ''
            };
            
            annotations.push(annotation);
            renderSegments();
            selectSegment(annotation.id);
            hideNewAnnotationDialog();
        }
        
        // Legacy function for shortcuts
        function newAnnotation() {
            showNewAnnotationDialog();
        }
        
        function renderSegments() {
            // Clear existing segments
            document.querySelectorAll('.timeline-segment').forEach(seg => seg.remove());
            
            annotations.forEach(ann => {
                const segment = createSegmentElement(ann);
                const track = document.querySelector(`.timeline-track[data-track="${ann.track}"]`);
                if (track) {
                    track.appendChild(segment);
                }
            });
        }
        
        function createSegmentElement(annotation) {
            const segment = document.createElement('div');
            segment.className = 'timeline-segment';
            segment.dataset.id = annotation.id;
            
            // Apply task-specific color
            const taskClass = annotation.task.toLowerCase().replace(/[\s\/]/g, '-');
            segment.classList.add(`segment-${taskClass}`);
            
            // Position and size
            segment.style.left = (annotation.start * timelineZoom) + 'px';
            segment.style.width = ((annotation.end - annotation.start) * timelineZoom) + 'px';
            
            // Label
            const label = document.createElement('div');
            label.className = 'segment-label';
            label.textContent = annotation.task;
            segment.appendChild(label);
            
            // Resize handles
            const leftHandle = document.createElement('div');
            leftHandle.className = 'segment-resize-handle left';
            segment.appendChild(leftHandle);
            
            const rightHandle = document.createElement('div');
            rightHandle.className = 'segment-resize-handle right';
            segment.appendChild(rightHandle);
            
            // Event handlers
            segment.addEventListener('mousedown', (e) => {
                console.log('Segment mousedown:', annotation.id, annotation.task);
                // Select immediately on mousedown for instant feedback
                selectSegment(annotation.id);
                
                if (e.target.classList.contains('segment-resize-handle')) {
                    startResize(e, annotation, e.target.classList.contains('left'));
                } else {
                    startDrag(e, annotation);
                }
            });
            
            segment.addEventListener('click', (e) => {
                console.log('Segment click:', annotation.id);
                // Stop propagation to prevent deselection
                e.stopPropagation();
            });
            
            segment.addEventListener('contextmenu', (e) => showContextMenu(e, annotation));
            
            return segment;
        }
        
        function startDrag(e, annotation) {
            let hasStartedDragging = false;
            const dragThreshold = 5; // pixels - must move at least this much to start dragging
            const startX = e.clientX;
            const startY = e.clientY;
            dragStartX = e.clientX;
            segmentStartPos = annotation.start;
            const originalTrack = annotation.track;
            
            const segment = document.querySelector(`.timeline-segment[data-id="${annotation.id}"]`);
            
            const handleDrag = (e) => {
                const deltaX = e.clientX - dragStartX;
                const deltaY = e.clientY - startY;
                
                // Only start dragging if moved beyond threshold
                if (!hasStartedDragging && Math.abs(e.clientX - startX) < dragThreshold) {
                    return;
                }
                
                if (!hasStartedDragging) {
                    hasStartedDragging = true;
                    isDragging = true;
                    segment.classList.add('dragging');
                }
                
                if (!isDragging) return;
                
                // Horizontal movement - time adjustment
                const deltaTime = deltaX / timelineZoom;
                const snappedTime = Math.round(deltaTime * 10) / 10; // Snap to 0.1s
                const newStart = Math.max(0, Math.min(videoDuration - (annotation.end - annotation.start), segmentStartPos + snappedTime));
                const duration = annotation.end - annotation.start;
                
                annotation.start = newStart;
                annotation.end = newStart + duration;
                
                segment.style.left = (annotation.start * timelineZoom) + 'px';
                
                // Vertical movement - track switching (only if second track is visible)
                if (hasSecondTrack && Math.abs(deltaY) > 30) {
                    const newTrack = deltaY > 0 ? 1 : 0;
                    if (newTrack !== annotation.track) {
                        annotation.track = newTrack;
                        // Re-render to move to new track
                        renderSegments();
                        // Re-select the segment after re-render
                        selectSegment(annotation.id);
                    }
                }
            };
            
            const stopDrag = () => {
                if (hasStartedDragging) {
                    isDragging = false;
                    segment.classList.remove('dragging');
                    updateProperties(annotation);
                }
                // No need to select here since we already selected on mousedown
                document.removeEventListener('mousemove', handleDrag);
                document.removeEventListener('mouseup', stopDrag);
            };
            
            document.addEventListener('mousemove', handleDrag);
            document.addEventListener('mouseup', stopDrag);
        }
        
        function startResize(e, annotation, isLeft) {
            e.stopPropagation();
            isResizing = true;
            dragStartX = e.clientX;
            const originalStart = annotation.start;
            const originalEnd = annotation.end;
            
            const segment = document.querySelector(`.timeline-segment[data-id="${annotation.id}"]`);
            
            const handleResize = (e) => {
                if (!isResizing) return;
                
                const deltaX = e.clientX - dragStartX;
                const deltaTime = deltaX / timelineZoom;
                // Snap to 0.1 second increments
                const snappedDelta = Math.round(deltaTime * 10) / 10;
                
                if (isLeft) {
                    const newStart = Math.max(0, Math.min(annotation.end - 0.5, originalStart + snappedDelta));
                    annotation.start = newStart;
                    segment.style.left = (annotation.start * timelineZoom) + 'px';
                    segment.style.width = ((annotation.end - annotation.start) * timelineZoom) + 'px';
                } else {
                    const newEnd = Math.max(annotation.start + 0.5, Math.min(videoDuration, originalEnd + snappedDelta));
                    annotation.end = newEnd;
                    segment.style.width = ((annotation.end - annotation.start) * timelineZoom) + 'px';
                }
            };
            
            const stopResize = () => {
                isResizing = false;
                
                // DON'T automatically jump when resizing - let user control playhead
                // if (isLeft) {
                //     video.currentTime = annotation.start;
                // } else {
                //     video.currentTime = annotation.end;
                // }
                
                document.removeEventListener('mousemove', handleResize);
                document.removeEventListener('mouseup', stopResize);
                updateProperties(annotation);
            };
            
            document.addEventListener('mousemove', handleResize);
            document.addEventListener('mouseup', stopResize);
        }
        
        function selectSegment(id) {
            // Clear previous selection
            document.querySelectorAll('.timeline-segment').forEach(seg => {
                seg.classList.remove('selected');
            });
            
            // Find the annotation
            const annotation = annotations.find(ann => ann.id === id);
            if (!annotation) return;
            
            // Select new segment
            const segment = document.querySelector(`.timeline-segment[data-id="${id}"]`);
            if (segment) {
                segment.classList.add('selected');
                selectedSegment = annotation;
                updateProperties(selectedSegment);
                
                // DON'T automatically jump to segment - let user control playhead
                // video.currentTime = selectedSegment.start;
            }
        }
        
        function deselectAll() {
            // Clear all selections
            document.querySelectorAll('.timeline-segment').forEach(seg => {
                seg.classList.remove('selected');
            });
            selectedSegment = null;
            
            // Clear properties panel
            document.getElementById('propertiesContent').innerHTML = `
                <div class="property-group">
                    <div class="property-label">No Selection</div>
                    <p style="font-size: 12px; color: #969696;">Select a segment to edit its properties</p>
                </div>
            `;
        }
        
        function updateProperties(annotation) {
            if (!annotation) return;
            
            const content = document.getElementById('propertiesContent');
            content.innerHTML = `
                <div class="property-group">
                    <div class="property-label">Timing</div>
                    <input type="number" class="property-input" value="${annotation.start.toFixed(2)}" 
                           onchange="updateAnnotation(${annotation.id}, 'start', parseFloat(this.value))"
                           step="0.1" min="0" max="${videoDuration}">
                    <input type="number" class="property-input" value="${annotation.end.toFixed(2)}" 
                           onchange="updateAnnotation(${annotation.id}, 'end', parseFloat(this.value))"
                           step="0.1" min="0" max="${videoDuration}" style="margin-top: 5px;">
                </div>
                
                <div class="property-group">
                    <div class="property-label">Task Type</div>
                    <select class="property-input" onchange="updateAnnotation(${annotation.id}, 'task', this.value)">
                        <option value="Finger Tapping" ${annotation.task === 'Finger Tapping' ? 'selected' : ''}>Finger Tapping</option>
                        <option value="Hand Opening/Closing" ${annotation.task === 'Hand Opening/Closing' ? 'selected' : ''}>Hand Opening/Closing</option>
                        <option value="Pronation-Supination" ${annotation.task === 'Pronation-Supination' ? 'selected' : ''}>Pronation-Supination</option>
                        <option value="Rest Tremor" ${annotation.task === 'Rest Tremor' ? 'selected' : ''}>Rest Tremor</option>
                        <option value="Postural Tremor" ${annotation.task === 'Postural Tremor' ? 'selected' : ''}>Postural Tremor</option>
                        <option value="Kinetic Tremor" ${annotation.task === 'Kinetic Tremor' ? 'selected' : ''}>Kinetic Tremor</option>
                        <option value="Gait" ${annotation.task === 'Gait' ? 'selected' : ''}>Gait</option>
                        <option value="Facial Expression" ${annotation.task === 'Facial Expression' ? 'selected' : ''}>Facial Expression</option>
                        <option value="Toe Tapping" ${annotation.task === 'Toe Tapping' ? 'selected' : ''}>Toe Tapping</option>
                        <option value="Leg Agility" ${annotation.task === 'Leg Agility' ? 'selected' : ''}>Leg Agility</option>
                        <option value="Speech" ${annotation.task === 'Speech' ? 'selected' : ''}>Speech</option>
                        <option value="Writing" ${annotation.task === 'Writing' ? 'selected' : ''}>Writing</option>
                    </select>
                </div>
                
                <div class="property-group">
                    <div class="property-label">Side</div>
                    <select class="property-input" onchange="updateAnnotation(${annotation.id}, 'side', this.value)">
                        <option value="bilateral" ${annotation.side === 'bilateral' ? 'selected' : ''}>Bilateral</option>
                        <option value="left" ${annotation.side === 'left' ? 'selected' : ''}>Left</option>
                        <option value="right" ${annotation.side === 'right' ? 'selected' : ''}>Right</option>
                        <option value="n/a" ${annotation.side === 'n/a' ? 'selected' : ''}>N/A</option>
                    </select>
                </div>
                
                <div class="property-group">
                    <div class="property-label">Severity (0-4)</div>
                    <input type="number" class="property-input" value="${annotation.severity}" 
                           onchange="updateAnnotation(${annotation.id}, 'severity', parseInt(this.value))"
                           min="0" max="4">
                </div>
                
                <div class="property-group">
                    <div class="property-label">Track</div>
                    <select class="property-input" onchange="updateAnnotation(${annotation.id}, 'track', parseInt(this.value))">
                        <option value="0" ${annotation.track === 0 ? 'selected' : ''}>Track 1</option>
                        ${hasSecondTrack ? `<option value="1" ${annotation.track === 1 ? 'selected' : ''}>Track 2</option>` : ''}
                    </select>
                </div>
                
                <div class="property-group">
                    <div class="property-label">Notes</div>
                    <textarea class="property-input" rows="4" 
                              onchange="updateAnnotation(${annotation.id}, 'notes', this.value)">${annotation.notes || ''}</textarea>
                </div>
            `;
        }
        
        function updateAnnotation(id, field, value) {
            const annotation = annotations.find(ann => ann.id === id);
            if (annotation) {
                annotation[field] = value;
                
                // Don't override track unless it doesn't exist
                if (annotation.track === undefined) {
                    annotation.track = 0;
                }
                
                renderSegments();
                selectSegment(id);
            }
        }
        
        function deleteSelected() {
            if (selectedSegment) {
                annotations = annotations.filter(ann => ann.id !== selectedSegment.id);
                selectedSegment = null;
                renderSegments();
                document.getElementById('propertiesContent').innerHTML = `
                    <div class="property-group">
                        <div class="property-label">No Selection</div>
                        <p style="font-size: 12px; color: #969696;">Select a segment to edit its properties</p>
                    </div>
                `;
            }
        }
        
        function duplicateSelected() {
            if (selectedSegment) {
                const duplicate = {
                    ...selectedSegment,
                    id: Date.now(),
                    start: selectedSegment.end,
                    end: Math.min(selectedSegment.end + (selectedSegment.end - selectedSegment.start), videoDuration)
                };
                annotations.push(duplicate);
                renderSegments();
                selectSegment(duplicate.id);
            }
        }
        
        function splitAtPlayhead() {
            if (selectedSegment && video.currentTime > selectedSegment.start && video.currentTime < selectedSegment.end) {
                const newSegment = {
                    ...selectedSegment,
                    id: Date.now(),
                    start: video.currentTime,
                    end: selectedSegment.end
                };
                selectedSegment.end = video.currentTime;
                annotations.push(newSegment);
                renderSegments();
            }
        }
        
        // Context menu
        function showContextMenu(e, annotation) {
            e.preventDefault();
            const menu = document.getElementById('contextMenu');
            menu.style.display = 'block';
            menu.style.left = e.clientX + 'px';
            menu.style.top = e.clientY + 'px';
            
            selectedSegment = annotation;
            selectSegment(annotation.id);
            
            document.addEventListener('click', hideContextMenu);
        }
        
        function hideContextMenu() {
            document.getElementById('contextMenu').style.display = 'none';
            document.removeEventListener('click', hideContextMenu);
        }
        
        function cutSegment() {
            if (selectedSegment) {
                clipboard = {...selectedSegment};
                deleteSelected();
            }
        }
        
        function copySegment() {
            if (selectedSegment) {
                clipboard = {...selectedSegment};
            }
        }
        
        function pasteSegment() {
            if (clipboard) {
                const newSegment = {
                    ...clipboard,
                    id: Date.now(),
                    start: video.currentTime,
                    end: video.currentTime + (clipboard.end - clipboard.start)
                };
                annotations.push(newSegment);
                renderSegments();
                selectSegment(newSegment.id);
            }
        }
        
        // Save/Load
        function saveProject() {
            const data = {
                annotations: annotations,
                videoDuration: videoDuration,
                timestamp: new Date().toISOString()
            };
            
            fetch('/save_project', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    showNotification('Project saved successfully');
                }
            });
        }
        
        function exportAnnotations() {
            window.location.href = '/export_annotations';
        }
        
        function showNotification(message) {
            // Simple notification (can be enhanced)
            const notification = document.createElement('div');
            notification.style.position = 'fixed';
            notification.style.bottom = '20px';
            notification.style.right = '20px';
            notification.style.background = '#007acc';
            notification.style.color = 'white';
            notification.style.padding = '10px 20px';
            notification.style.borderRadius = '4px';
            notification.style.zIndex = '1000';
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }
        
        // Keyboard shortcuts
        function showShortcuts() {
            document.getElementById('shortcutsModal').style.display = 'block';
        }
        
        function hideShortcuts() {
            document.getElementById('shortcutsModal').style.display = 'none';
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    togglePlayPause();
                    break;
                case 's':
                    if (!e.metaKey && !e.ctrlKey) {
                        splitAtPlayhead();
                    }
                    break;
                case 'Delete':
                case 'Backspace':
                    deleteSelected();
                    break;
                case 'd':
                    if (e.metaKey || e.ctrlKey) {
                        e.preventDefault();
                        duplicateSelected();
                    }
                    break;
                case 's':
                    if (e.metaKey || e.ctrlKey) {
                        e.preventDefault();
                        saveProject();
                    }
                    break;
                case 'm':
                    newAnnotation();
                    break;
            }
        });
        
        // Load annotations on start
        fetch('/get_annotations')
            .then(response => response.json())
            .then(data => {
                annotations = data.annotations || [];
                renderSegments();
            });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    video_url = url_for('serve_video') if current_video_path else ''
    return render_template_string(HTML_TEMPLATE, video_url=video_url)

@app.route('/serve_video')
def serve_video():
    if current_video_path and os.path.exists(current_video_path):
        return send_file(current_video_path, mimetype='video/mp4')
    return '', 404

@app.route('/get_annotations')
def get_annotations():
    return jsonify({'annotations': annotations})

@app.route('/save_project', methods=['POST'])
def save_project():
    global annotations
    data = request.json
    annotations = data.get('annotations', [])
    
    # Save to file
    filename = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/export_annotations')
def export_annotations():
    data = {
        'video': current_video_path,
        'annotations': annotations,
        'created': datetime.now().isoformat()
    }
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, temp_file, indent=2)
    temp_file.close()
    
    return send_file(temp_file.name, as_attachment=True, download_name='annotations.json')

def start_server(video_path):
    global current_video_path, video_info
    current_video_path = video_path
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    video_info = {
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    
    print(f"Starting Video Editor Annotation Server")
    print(f"Video: {video_path}")
    print(f"Duration: {video_info['duration']:.1f}s")
    print("Open http://localhost:5555 in your browser")
    app.run(host='0.0.0.0', port=5555, debug=False)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        start_server(sys.argv[1])
    else:
        print("Usage: python video_editor_annotation_server.py <video_path>")