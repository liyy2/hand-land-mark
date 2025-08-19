#!/usr/bin/env python
"""
Enhanced Video Annotation Server with Heatmap, Labels, and Edit Capabilities
Includes movement analysis heatmap and better annotation management
"""

from flask import Flask, render_template_string, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile
import shutil
import base64
from io import BytesIO
from unified_landmark_detector import UnifiedLandmarkDetector
import sys

app = Flask(__name__)
CORS(app)

# Store annotations and analysis data in memory
annotations = []
current_video_path = None
original_video_path = None  # Store the original video path
original_video_name = None  # Store the original video name
landmark_data = None
heatmap_cache = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Video Annotator - Advanced Edition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #151521 100%);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        h1 {
            color: #ffffff;
            margin-bottom: 30px;
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .video-section {
            background: #2a2a3e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        video {
            width: 100%;
            max-height: 500px;
            border-radius: 8px;
            background: #000;
        }
        
        .time-display {
            font-size: 48px;
            font-weight: bold;
            color: #4ade80;
            font-family: 'Monaco', 'Courier New', monospace;
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 8px;
            margin: 20px 0;
            text-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
        }
        
        .control-panel {
            background: #2a2a3e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .control-panel h3 {
            color: #ffffff;
            margin-bottom: 15px;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        button {
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }
        
        .btn-info {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 10px;
            margin: 8px 0;
            background: #1a1a2e;
            border: 2px solid #3a3a4e;
            color: #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        label {
            display: block;
            margin-top: 12px;
            margin-bottom: 4px;
            color: #a0a0a0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .capture-buttons {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        
        .segment-times {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        
        .keyboard-hint {
            background: linear-gradient(135deg, #3a3a4e 0%, #2a2a3e 100%);
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 12px;
            border: 1px solid #4a4a5e;
        }
        
        .keyboard-hint strong {
            color: #667eea;
        }
        
        .annotations-table {
            background: #2a2a3e;
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background: #3a3a4e;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #ffffff;
            border-bottom: 2px solid #4a4a5e;
        }
        
        td {
            padding: 10px;
            border-bottom: 1px solid #3a3a4e;
        }
        
        tr:hover {
            background: #3a3a4e;
        }
        
        .annotation-label {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .label-finger-tapping { background: #667eea; }
        .label-tremor { background: #ef4444; }
        .label-hand-opening { background: #10b981; }
        .label-pronation { background: #f59e0b; }
        .label-gait { background: #8b5cf6; }
        .label-facial { background: #ec4899; }
        
        .severity-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }
        
        .severity-0 { background: #10b981; color: white; }
        .severity-1 { background: #84cc16; color: white; }
        .severity-2 { background: #f59e0b; color: white; }
        .severity-3 { background: #f97316; color: white; }
        .severity-4 { background: #ef4444; color: white; }
        
        .heatmap-section {
            background: #2a2a3e;
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .heatmap-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .heatmap-image {
            width: 100%;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        .edit-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .edit-modal.active {
            display: flex;
        }
        
        .edit-modal-content {
            background: #2a2a3e;
            padding: 30px;
            border-radius: 12px;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .modal-header h3 {
            color: #ffffff;
            margin: 0;
        }
        
        .close-modal {
            background: transparent;
            color: #ef4444;
            font-size: 24px;
            cursor: pointer;
            border: none;
            padding: 0;
        }
        
        .action-buttons {
            display: flex;
            gap: 8px;
        }
        
        .action-buttons button {
            padding: 6px 12px;
            font-size: 12px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #3a3a4e 0%, #2a2a3e 100%);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #4a4a5e;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 12px;
            color: #a0a0a0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 5px;
        }
        
        .timeline-bar {
            position: relative;
            height: 40px;
            background: #1a1a2e;
            border-radius: 8px;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .timeline-segment {
            position: absolute;
            height: 100%;
            border-radius: 4px;
            opacity: 0.8;
            cursor: pointer;
            transition: opacity 0.3s ease;
        }
        
        .timeline-segment:hover {
            opacity: 1;
        }
        
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Enhanced Video Annotation Studio</h1>
        
        <div class="main-grid">
            <div class="video-section">
                <video id="videoPlayer" controls>
                    <source src="{{ video_url }}" type="video/mp4">
                </video>
                
                <div class="time-display" id="currentTime">0:00.0</div>
                
                <!-- Timeline visualization -->
                <div class="timeline-bar" id="timeline"></div>
                
                <div class="capture-buttons">
                    <button class="btn-primary" onclick="captureTime()">📍 Capture</button>
                    <button class="btn-success" onclick="setAsStart()">⬅️ Set Start</button>
                    <button class="btn-warning" onclick="setAsEnd()">➡️ Set End</button>
                </div>
            </div>
            
            <div>
                <div class="control-panel">
                    <h3>⏱️ Segment Control</h3>
                    
                    <div class="segment-times">
                        <div>
                            <label>Start Time (sec)</label>
                            <input type="number" id="startTime" step="0.1" value="0">
                        </div>
                        <div>
                            <label>End Time (sec)</label>
                            <input type="number" id="endTime" step="0.1" value="5">
                        </div>
                    </div>
                    
                    <button class="btn-info" onclick="quickSegment()" style="width: 100%;">
                        ⚡ Quick 5-second Segment
                    </button>
                    
                    <label>Task Type</label>
                    <select id="taskType">
                        <option value="Finger Tapping">Finger Tapping</option>
                        <option value="Hand Opening/Closing">Hand Opening/Closing</option>
                        <option value="Pronation-Supination">Pronation-Supination</option>
                        <option value="Rest Tremor">Rest Tremor</option>
                        <option value="Postural Tremor">Postural Tremor</option>
                        <option value="Kinetic Tremor">Kinetic Tremor</option>
                        <option value="Gait">Gait</option>
                        <option value="Facial Expression">Facial Expression</option>
                        <option value="Finger to Nose">Finger to Nose</option>
                        <option value="Heel to Shin">Heel to Shin</option>
                        <option value="Toe Tapping">Toe Tapping</option>
                        <option value="Leg Agility">Leg Agility</option>
                        <option value="Arising from Chair">Arising from Chair</option>
                        <option value="Postural Stability">Postural Stability</option>
                        <option value="Speech">Speech</option>
                        <option value="Writing">Writing</option>
                        <option value="Other">Other</option>
                    </select>
                    
                    <label>Side</label>
                    <select id="side">
                        <option value="bilateral">Bilateral</option>
                        <option value="left">Left</option>
                        <option value="right">Right</option>
                        <option value="n/a">N/A</option>
                    </select>
                    
                    <label>Severity (UPDRS 0-4)</label>
                    <input type="number" id="severity" min="0" max="4" value="0">
                    
                    <label>Notes</label>
                    <textarea id="notes" rows="3" placeholder="Optional clinical notes..."></textarea>
                    
                    <button class="btn-success" onclick="addAnnotation()" style="width: 100%; margin-top: 15px;">
                        ➕ Add Annotation
                    </button>
                    
                    <div class="keyboard-hint">
                        <strong>Keyboard Shortcuts:</strong><br>
                        Space: Play/Pause | C: Capture | S: Set Start<br>
                        E: Set End | Q: Quick Segment | A: Add Annotation
                    </div>
                </div>
                
                <!-- Statistics -->
                <div class="control-panel" style="margin-top: 20px;">
                    <h3>📊 Session Statistics</h3>
                    <div class="stats-grid" id="statsGrid">
                        <div class="stat-card">
                            <div class="stat-value" id="totalAnnotations">0</div>
                            <div class="stat-label">Total Annotations</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="totalDuration">0s</div>
                            <div class="stat-label">Annotated Time</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="avgSeverity">0.0</div>
                            <div class="stat-label">Avg Severity</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="taskCount">0</div>
                            <div class="stat-label">Unique Tasks</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Heatmap Section -->
        <div class="heatmap-section">
            <h3>🔥 Movement Heatmap Analysis</h3>
            <div class="heatmap-controls">
                <button class="btn-primary" onclick="generateHeatmap('hand')">Generate Hand Heatmap</button>
                <button class="btn-info" onclick="generateHeatmap('trajectory')">Generate Trajectory Map</button>
                <button class="btn-warning" onclick="generateHeatmap('velocity')">Generate Velocity Map</button>
                <select id="heatmapHand" style="width: 150px;">
                    <option value="both">Both Hands</option>
                    <option value="left">Left Hand</option>
                    <option value="right">Right Hand</option>
                </select>
                <select id="heatmapLandmark" style="width: 150px;">
                    <option value="8">Index Finger Tip</option>
                    <option value="4">Thumb Tip</option>
                    <option value="12">Middle Finger Tip</option>
                    <option value="0">Wrist</option>
                </select>
            </div>
            <div id="heatmapContainer"></div>
        </div>
        
        <!-- Annotations Table -->
        <div class="annotations-table">
            <h3>📋 Annotations</h3>
            <div style="margin-bottom: 20px;">
                <input type="text" id="searchFilter" placeholder="Search annotations..." style="width: 300px;">
                <select id="taskFilter" style="width: 200px; margin-left: 10px;">
                    <option value="">All Tasks</option>
                    <option value="Finger Tapping">Finger Tapping</option>
                    <option value="Tremor">Tremor Tasks</option>
                    <option value="Gait">Gait Tasks</option>
                </select>
            </div>
            
            <table id="annotationsTable">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Time Range</th>
                        <th>Duration</th>
                        <th>Task</th>
                        <th>Side</th>
                        <th>Severity</th>
                        <th>Notes</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="annotationsBody">
                </tbody>
            </table>
            
            <div style="margin-top: 20px; display: flex; gap: 10px;">
                <button class="btn-primary" onclick="saveAnnotations()">💾 Save All</button>
                <button class="btn-success" onclick="exportJSON()">📥 Export JSON</button>
                <button class="btn-info" onclick="exportCSV()">📊 Export CSV</button>
                <button class="btn-warning" onclick="importAnnotations()">📤 Import</button>
                <button class="btn-danger" onclick="clearAnnotations()">🗑️ Clear All</button>
            </div>
        </div>
        
        <!-- Legend -->
        <div class="annotations-table">
            <h3>🎨 Task Color Legend</h3>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #667eea;"></div>
                    <span>Finger Tapping</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ef4444;"></div>
                    <span>Tremor</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #10b981;"></div>
                    <span>Hand Opening</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f59e0b;"></div>
                    <span>Pronation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #8b5cf6;"></div>
                    <span>Gait</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ec4899;"></div>
                    <span>Facial</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Edit Modal -->
    <div id="editModal" class="edit-modal">
        <div class="edit-modal-content">
            <div class="modal-header">
                <h3>Edit Annotation</h3>
                <button class="close-modal" onclick="closeEditModal()">&times;</button>
            </div>
            <div id="editModalBody">
                <!-- Edit form will be inserted here -->
            </div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('videoPlayer');
        const timeDisplay = document.getElementById('currentTime');
        let capturedTime = 0;
        let editingIndex = -1;
        
        // Color mapping for tasks
        const taskColors = {
            'Finger Tapping': '#667eea',
            'Hand Opening/Closing': '#10b981',
            'Pronation-Supination': '#f59e0b',
            'Rest Tremor': '#ef4444',
            'Postural Tremor': '#ef4444',
            'Kinetic Tremor': '#ef4444',
            'Gait': '#8b5cf6',
            'Facial Expression': '#ec4899',
            'Other': '#6b7280'
        };
        
        // Update time display continuously
        video.addEventListener('timeupdate', () => {
            const time = video.currentTime;
            const minutes = Math.floor(time / 60);
            const seconds = (time % 60).toFixed(1);
            timeDisplay.textContent = `${minutes}:${seconds.padStart(4, '0')}`;
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key.toLowerCase()) {
                case ' ':
                    e.preventDefault();
                    video.paused ? video.play() : video.pause();
                    break;
                case 'c':
                    captureTime();
                    break;
                case 's':
                    setAsStart();
                    break;
                case 'e':
                    setAsEnd();
                    break;
                case 'q':
                    quickSegment();
                    break;
                case 'a':
                    addAnnotation();
                    break;
            }
        });
        
        function captureTime() {
            capturedTime = video.currentTime;
            timeDisplay.style.background = 'linear-gradient(135deg, #065f46 0%, #047857 100%)';
            setTimeout(() => { 
                timeDisplay.style.background = 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)'; 
            }, 300);
            return capturedTime;
        }
        
        function setAsStart() {
            const time = captureTime();
            document.getElementById('startTime').value = time.toFixed(1);
        }
        
        function setAsEnd() {
            const time = captureTime();
            document.getElementById('endTime').value = time.toFixed(1);
        }
        
        function quickSegment() {
            const time = captureTime();
            document.getElementById('startTime').value = Math.max(0, time - 2.5).toFixed(1);
            document.getElementById('endTime').value = (time + 2.5).toFixed(1);
        }
        
        function addAnnotation() {
            const annotation = {
                start: parseFloat(document.getElementById('startTime').value),
                end: parseFloat(document.getElementById('endTime').value),
                task: document.getElementById('taskType').value,
                side: document.getElementById('side').value,
                severity: parseInt(document.getElementById('severity').value),
                notes: document.getElementById('notes').value
            };
            
            // Validate
            if (annotation.end <= annotation.start) {
                alert('End time must be after start time!');
                return;
            }
            
            fetch('/add_annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotation)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadAnnotations();
                    updateStats();
                    updateTimeline();
                    // Clear form
                    document.getElementById('notes').value = '';
                }
            });
        }
        
        function loadAnnotations() {
            fetch('/get_annotations')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('annotationsBody');
                    tbody.innerHTML = '';
                    
                    // Apply filters
                    const searchTerm = document.getElementById('searchFilter').value.toLowerCase();
                    const taskFilter = document.getElementById('taskFilter').value;
                    
                    let filteredAnnotations = data.annotations;
                    if (searchTerm) {
                        filteredAnnotations = filteredAnnotations.filter(ann => 
                            ann.task.toLowerCase().includes(searchTerm) ||
                            ann.notes.toLowerCase().includes(searchTerm)
                        );
                    }
                    if (taskFilter) {
                        if (taskFilter === 'Tremor') {
                            filteredAnnotations = filteredAnnotations.filter(ann => 
                                ann.task.includes('Tremor')
                            );
                        } else {
                            filteredAnnotations = filteredAnnotations.filter(ann => 
                                ann.task === taskFilter
                            );
                        }
                    }
                    
                    filteredAnnotations.forEach((ann, idx) => {
                        const row = tbody.insertRow();
                        const taskClass = ann.task.toLowerCase().replace(/[\/\s]/g, '-');
                        const labelClass = `annotation-label label-${taskClass}`;
                        
                        row.innerHTML = `
                            <td>${idx}</td>
                            <td>${formatTime(ann.start)} - ${formatTime(ann.end)}</td>
                            <td>${(ann.end - ann.start).toFixed(1)}s</td>
                            <td><span class="${labelClass}">${ann.task}</span></td>
                            <td>${ann.side}</td>
                            <td><span class="severity-badge severity-${ann.severity}">${ann.severity}</span></td>
                            <td>${ann.notes || '-'}</td>
                            <td>
                                <div class="action-buttons">
                                    <button class="btn-info" onclick="jumpToSegment(${ann.start}, ${ann.end})">▶️</button>
                                    <button class="btn-warning" onclick="editAnnotation(${idx})">✏️</button>
                                    <button class="btn-danger" onclick="deleteAnnotation(${idx})">🗑️</button>
                                </div>
                            </td>
                        `;
                    });
                    
                    updateStats();
                    updateTimeline();
                });
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(1);
            return `${mins}:${secs.padStart(4, '0')}`;
        }
        
        function jumpToSegment(start, end) {
            video.currentTime = start;
            video.play();
            // Optionally pause at the end
            setTimeout(() => {
                if (video.currentTime >= end) {
                    video.pause();
                }
            }, (end - start) * 1000);
        }
        
        function editAnnotation(idx) {
            fetch('/get_annotation/' + idx)
                .then(response => response.json())
                .then(data => {
                    if (data.annotation) {
                        editingIndex = idx;
                        showEditModal(data.annotation);
                    }
                });
        }
        
        function showEditModal(annotation) {
            const modal = document.getElementById('editModal');
            const body = document.getElementById('editModalBody');
            
            body.innerHTML = `
                <label>Start Time (sec)</label>
                <input type="number" id="editStartTime" step="0.1" value="${annotation.start}">
                
                <label>End Time (sec)</label>
                <input type="number" id="editEndTime" step="0.1" value="${annotation.end}">
                
                <label>Task Type</label>
                <select id="editTaskType">
                    ${document.getElementById('taskType').innerHTML}
                </select>
                
                <label>Side</label>
                <select id="editSide">
                    ${document.getElementById('side').innerHTML}
                </select>
                
                <label>Severity (UPDRS 0-4)</label>
                <input type="number" id="editSeverity" min="0" max="4" value="${annotation.severity}">
                
                <label>Notes</label>
                <textarea id="editNotes" rows="3">${annotation.notes || ''}</textarea>
                
                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button class="btn-success" onclick="saveEdit()">💾 Save Changes</button>
                    <button class="btn-danger" onclick="closeEditModal()">Cancel</button>
                </div>
            `;
            
            // Set the correct values
            document.getElementById('editTaskType').value = annotation.task;
            document.getElementById('editSide').value = annotation.side;
            
            modal.classList.add('active');
        }
        
        function closeEditModal() {
            document.getElementById('editModal').classList.remove('active');
            editingIndex = -1;
        }
        
        function saveEdit() {
            const updatedAnnotation = {
                start: parseFloat(document.getElementById('editStartTime').value),
                end: parseFloat(document.getElementById('editEndTime').value),
                task: document.getElementById('editTaskType').value,
                side: document.getElementById('editSide').value,
                severity: parseInt(document.getElementById('editSeverity').value),
                notes: document.getElementById('editNotes').value
            };
            
            fetch('/update_annotation/' + editingIndex, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(updatedAnnotation)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    closeEditModal();
                    loadAnnotations();
                }
            });
        }
        
        function deleteAnnotation(idx) {
            if (confirm('Delete this annotation?')) {
                fetch(`/delete_annotation/${idx}`, {method: 'DELETE'})
                    .then(() => loadAnnotations());
            }
        }
        
        function clearAnnotations() {
            if (confirm('Clear all annotations? This cannot be undone!')) {
                fetch('/clear_annotations', {method: 'POST'})
                    .then(() => loadAnnotations());
            }
        }
        
        function saveAnnotations() {
            fetch('/save_annotations', {method: 'POST'})
                .then(response => response.json())
                .then(data => alert(data.message));
        }
        
        function exportJSON() {
            window.location.href = '/export_annotations';
        }
        
        function exportCSV() {
            window.location.href = '/export_csv';
        }
        
        function importAnnotations() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (e) => {
                const file = e.target.files[0];
                const formData = new FormData();
                formData.append('file', file);
                
                fetch('/import_annotations', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadAnnotations();
                        alert('Annotations imported successfully!');
                    }
                });
            };
            input.click();
        }
        
        function updateStats() {
            fetch('/get_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalAnnotations').textContent = data.total;
                    document.getElementById('totalDuration').textContent = data.duration + 's';
                    document.getElementById('avgSeverity').textContent = data.avg_severity.toFixed(1);
                    document.getElementById('taskCount').textContent = data.unique_tasks;
                });
        }
        
        function updateTimeline() {
            fetch('/get_annotations')
                .then(response => response.json())
                .then(data => {
                    const timeline = document.getElementById('timeline');
                    timeline.innerHTML = '';
                    
                    const videoDuration = video.duration || 100;
                    
                    data.annotations.forEach(ann => {
                        const segment = document.createElement('div');
                        segment.className = 'timeline-segment';
                        segment.style.left = `${(ann.start / videoDuration) * 100}%`;
                        segment.style.width = `${((ann.end - ann.start) / videoDuration) * 100}%`;
                        segment.style.background = taskColors[ann.task] || '#6b7280';
                        segment.title = `${ann.task} (${formatTime(ann.start)} - ${formatTime(ann.end)})`;
                        segment.onclick = () => jumpToSegment(ann.start, ann.end);
                        timeline.appendChild(segment);
                    });
                });
        }
        
        function generateHeatmap(type) {
            const hand = document.getElementById('heatmapHand').value;
            const landmark = document.getElementById('heatmapLandmark').value;
            
            fetch(`/generate_heatmap?type=${type}&hand=${hand}&landmark=${landmark}`)
                .then(response => response.json())
                .then(data => {
                    if (data.heatmap) {
                        const container = document.getElementById('heatmapContainer');
                        container.innerHTML = `<img src="data:image/png;base64,${data.heatmap}" class="heatmap-image">`;
                    }
                });
        }
        
        // Filter handlers
        document.getElementById('searchFilter').addEventListener('input', loadAnnotations);
        document.getElementById('taskFilter').addEventListener('change', loadAnnotations);
        
        // Load annotations on start
        loadAnnotations();
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

@app.route('/add_annotation', methods=['POST'])
def add_annotation():
    annotation = request.json
    annotation['timestamp'] = datetime.now().isoformat()
    annotations.append(annotation)
    return jsonify({'success': True})

@app.route('/get_annotations')
def get_annotations():
    return jsonify({'annotations': annotations})

@app.route('/get_annotation/<int:idx>')
def get_annotation(idx):
    if 0 <= idx < len(annotations):
        return jsonify({'annotation': annotations[idx]})
    return jsonify({'error': 'Not found'}), 404

@app.route('/update_annotation/<int:idx>', methods=['PUT'])
def update_annotation(idx):
    if 0 <= idx < len(annotations):
        annotation = request.json
        annotation['timestamp'] = annotations[idx].get('timestamp', datetime.now().isoformat())
        annotation['modified'] = datetime.now().isoformat()
        annotations[idx] = annotation
        return jsonify({'success': True})
    return jsonify({'error': 'Not found'}), 404

@app.route('/delete_annotation/<int:idx>', methods=['DELETE'])
def delete_annotation(idx):
    if 0 <= idx < len(annotations):
        annotations.pop(idx)
    return jsonify({'success': True})

@app.route('/clear_annotations', methods=['POST'])
def clear_annotations():
    global annotations
    annotations = []
    return jsonify({'success': True})

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    # Use original video name for the annotation file
    if original_video_name:
        base_name = os.path.splitext(original_video_name)[0]
        filename = f"{base_name}_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        filename = f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'video': original_video_path if original_video_path else current_video_path,
            'annotations': annotations,
            'created': datetime.now().isoformat()
        }, f, indent=2)
    return jsonify({'success': True, 'message': f'Saved to {filename}'})

@app.route('/export_annotations')
def export_annotations():
    # Use original video name for the download file
    if original_video_name:
        base_name = os.path.splitext(original_video_name)[0]
        download_name = f"{base_name}_annotations.json"
    else:
        download_name = 'annotations.json'
    
    data = {
        'video': original_video_path if original_video_path else current_video_path,
        'annotations': annotations,
        'created': datetime.now().isoformat()
    }
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, temp_file, indent=2)
    temp_file.close()
    return send_file(temp_file.name, as_attachment=True, download_name=download_name)

@app.route('/export_csv')
def export_csv():
    """Export annotations as CSV for analysis"""
    df_data = []
    for ann in annotations:
        df_data.append({
            'start_time': ann['start'],
            'end_time': ann['end'],
            'duration': ann['end'] - ann['start'],
            'task': ann['task'],
            'side': ann['side'],
            'severity': ann['severity'],
            'notes': ann.get('notes', '')
        })
    
    df = pd.DataFrame(df_data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return send_file(temp_file.name, as_attachment=True, download_name='annotations.csv')

@app.route('/import_annotations', methods=['POST'])
def import_annotations():
    """Import annotations from JSON file"""
    global annotations
    try:
        file = request.files['file']
        data = json.load(file)
        annotations = data.get('annotations', [])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stats')
def get_stats():
    """Get annotation statistics"""
    if not annotations:
        return jsonify({
            'total': 0,
            'duration': 0,
            'avg_severity': 0,
            'unique_tasks': 0
        })
    
    total_duration = sum(ann['end'] - ann['start'] for ann in annotations)
    avg_severity = sum(ann['severity'] for ann in annotations) / len(annotations)
    unique_tasks = len(set(ann['task'] for ann in annotations))
    
    return jsonify({
        'total': len(annotations),
        'duration': round(total_duration, 1),
        'avg_severity': avg_severity,
        'unique_tasks': unique_tasks
    })

@app.route('/generate_heatmap')
def generate_heatmap():
    """Generate movement heatmap from landmark data"""
    heatmap_type = request.args.get('type', 'hand')
    hand_filter = request.args.get('hand', 'both')
    landmark_id = int(request.args.get('landmark', 8))
    
    # Check if we have cached landmarks or need to process
    cache_key = f"{heatmap_type}_{hand_filter}_{landmark_id}"
    if cache_key in heatmap_cache:
        return jsonify({'heatmap': heatmap_cache[cache_key]})
    
    try:
        # For demo purposes, create a sample heatmap
        # In production, this would process actual landmark data
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if heatmap_type == 'hand':
            # Create sample data for hand position heatmap
            x = np.random.randn(1000) * 100 + 320
            y = np.random.randn(1000) * 100 + 240
            
            # Create heatmap
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
            ax.set_title(f'Movement Heatmap - {hand_filter.capitalize()} Hand', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            plt.colorbar(im, ax=ax, label='Frequency')
            
        elif heatmap_type == 'trajectory':
            # Create sample trajectory
            t = np.linspace(0, 10, 500)
            x = 320 + 100 * np.sin(t) + np.random.randn(500) * 5
            y = 240 + 100 * np.cos(t) + np.random.randn(500) * 5
            
            # Plot trajectory with color gradient
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis', linewidth=2)
            lc.set_array(t)
            ax.add_collection(lc)
            ax.autoscale()
            
            ax.set_title(f'Movement Trajectory - {hand_filter.capitalize()} Hand', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            plt.colorbar(lc, ax=ax, label='Time (s)')
            
        elif heatmap_type == 'velocity':
            # Create sample velocity heatmap
            t = np.linspace(0, 60, 600)
            velocity = np.abs(np.sin(t * 2) * 50 + np.random.randn(600) * 10)
            
            # Create 2D representation
            velocity_2d = velocity.reshape(20, 30)
            
            im = ax.imshow(velocity_2d, cmap='plasma', aspect='auto')
            ax.set_title(f'Velocity Heatmap - {hand_filter.capitalize()} Hand', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Window')
            ax.set_ylabel('Segment')
            plt.colorbar(im, ax=ax, label='Velocity (pixels/s)')
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Cache the result
        heatmap_cache[cache_key] = img_base64
        
        return jsonify({'heatmap': img_base64})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_landmarks(video_path):
    """Process video to extract landmarks for heatmap generation"""
    global landmark_data
    
    try:
        # Initialize detector
        detector = UnifiedLandmarkDetector(
            extract_hands=True,
            extract_face=False,
            max_num_hands=2,
            hand_detection_confidence=0.3,
            hand_tracking_confidence=0.3
        )
        
        # Process video frames
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            _, landmarks_dict = detector.process_frame(frame, frame_idx)
            if landmarks_dict:
                landmarks_list.append(landmarks_dict)
            
            frame_idx += 1
            if frame_idx > 300:  # Limit for demo
                break
        
        cap.release()
        landmark_data = landmarks_list
        
    except Exception as e:
        print(f"Error processing video: {e}")
        landmark_data = None

def convert_to_720p(input_path, output_dir=None):
    """Convert video to 720p resolution for better performance"""
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video resolution: {width}x{height}")
    
    # Check if conversion is needed
    if height <= 720:
        print(f"Video is already {height}p, no conversion needed")
        cap.release()
        return input_path
    
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 16/9:  # Wider than 16:9
        new_width = 1280
        new_height = int(1280 / aspect_ratio)
    else:  # Taller than or equal to 16:9
        new_height = 720
        new_width = int(720 * aspect_ratio)
    
    # Ensure dimensions are even (required for some codecs)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    print(f"Converting to: {new_width}x{new_height}")
    
    # Create output path
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_720p_")
    
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_720p{ext}")
    
    # Setup video writer with H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # If H264 fails, try mp4v
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Process frames
    frame_count = 0
    print("Converting video to 720p...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            sys.stdout.write(f"\rProgress: {progress:.1f}%")
            sys.stdout.flush()
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nConversion complete! Saved to: {output_path}")
    return output_path

def start_server(video_path):
    global current_video_path, original_video_path, original_video_name
    
    # Store original video information
    original_video_path = os.path.abspath(video_path)
    original_video_name = os.path.basename(video_path)
    
    # Convert video to 720p if needed
    print("Checking video resolution...")
    converted_path = convert_to_720p(video_path)
    current_video_path = converted_path
    
    # Process video for landmarks (optional, for heatmap)
    # process_video_landmarks(current_video_path)
    
    print(f"Starting enhanced annotation server")
    print(f"Original video: {original_video_path}")
    print(f"Using processed video: {current_video_path}")
    print("Open http://localhost:5555 in your browser")
    app.run(host='0.0.0.0', port=5555, debug=False)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        start_server(sys.argv[1])
    else:
        print("Usage: python enhanced_annotation_server.py <video_path>")