#!/usr/bin/env python
"""
Professional Video Editor-Style Annotation Server
Timeline-based interface similar to Final Cut Pro / Premiere Pro
"""

from flask import Flask, render_template_string, request, jsonify, send_file, url_for, make_response, redirect
from flask_cors import CORS
import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import argparse
from tqdm import tqdm
import sys
import time
import subprocess
import threading
import re

app = Flask(__name__)
CORS(app)

# Flask configuration for file uploads
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size (Disabled to allow unlimited uploads)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix="video_uploads_")

# Store annotations and video info
annotations = []
current_video_path = None
original_video_path = None  # Store the original video path
original_video_name = None  # Store the original video name
video_info = {}
video_ready = False  # Track if video is processed and ready
upload_folder = tempfile.mkdtemp(prefix="video_uploads_")
processing_status = {"status": "idle", "progress": 0, "message": ""}
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
video_version = 0  # Incremented whenever a new video is ready
def get_git_version():
    """Return the current short git commit hash for display."""
    try:
        result = subprocess.run(
            ['git', '-C', REPO_ROOT, 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"

GIT_VERSION = get_git_version()


def reset_video_state():
    """Clear server-side state before loading a new video."""
    global annotations, current_video_path, original_video_path, original_video_name
    global video_info, video_ready, video_version
    annotations = []
    current_video_path = None
    original_video_path = None
    original_video_name = None
    video_info = {}
    video_ready = False
    video_version = int(time.time() * 1000)

UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Annotation Editor - Upload</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #151521 100%);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .upload-container {
            max-width: 600px;
            width: 100%;
            background: #2a2a3e;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 16px 48px rgba(0,0,0,0.4);
        }
        
        h1 {
            color: #ffffff;
            margin-bottom: 10px;
            font-size: 32px;
            text-align: center;
        }
        
        .subtitle {
            text-align: center;
            color: #a0a0a0;
            margin-bottom: 40px;
            font-size: 14px;
        }
        
        .upload-area {
            border: 3px dashed #007acc;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #1a1a2e;
            margin-bottom: 30px;
        }
        
        .upload-area:hover {
            border-color: #1a86d3;
            background: #222238;
        }
        
        .upload-area.dragover {
            border-color: #10b981;
            background: #1a2e23;
        }
        
        .upload-icon {
            font-size: 64px;
            margin-bottom: 15px;
        }
        
        .upload-text {
            font-size: 18px;
            color: #e0e0e0;
            margin-bottom: 10px;
        }
        
        .upload-hint {
            font-size: 13px;
            color: #a0a0a0;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .divider {
            text-align: center;
            margin: 30px 0;
            position: relative;
        }
        
        .divider::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: #3a3a4e;
        }
        
        .divider span {
            background: #2a2a3e;
            padding: 0 15px;
            position: relative;
            color: #a0a0a0;
            font-size: 13px;
        }
        
        .path-input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #a0a0a0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            background: #1a1a2e;
            border: 2px solid #3a3a4e;
            color: #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #007acc;
            box-shadow: 0 0 0 3px rgba(0, 122, 204, 0.1);
        }
        
        button {
            width: 100%;
            padding: 14px 20px;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #007acc 0%, #0056a3 100%);
            color: white;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 122, 204, 0.4);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-container {
            display: none;
            margin-top: 30px;
            padding: 20px;
            background: #1a1a2e;
            border-radius: 8px;
        }
        
        .progress-container.active {
            display: block;
        }
        
        .progress-bar-container {
            width: 100%;
            height: 8px;
            background: #3a3a4e;
            border-radius: 4px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #007acc 0%, #10b981 100%);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 4px;
        }
        
        .progress-text {
            text-align: center;
            color: #a0a0a0;
            font-size: 13px;
            margin-bottom: 10px;
        }
        
        .progress-percentage {
            text-align: center;
            color: #007acc;
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .status-message {
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 14px;
        }
        
        .status-success {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }
        
        .status-error {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid #ef4444;
        }
        
        .top-controls {
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 100;
        }

        .restart-btn {
            width: auto;
            padding: 10px 20px;
            background: #2a2a3e;
            border: 1px solid #3a3a4e;
            color: #a0a0a0;
            font-size: 13px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .restart-btn:hover {
            background: #3a3a4e;
            color: #ffffff;
            border-color: #ff9800;
        }

        .update-btn, .feedback-btn {
            width: auto;
            padding: 10px 20px;
            background: #2a2a3e;
            border: 1px solid #3a3a4e;
            color: #a0a0a0;
            font-size: 13px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .update-btn:hover, .feedback-btn:hover {
            background: #3a3a4e;
            color: #ffffff;
        }
        
        .update-btn:hover {
            border-color: #007acc;
        }
        
        .feedback-btn:hover {
            border-color: #10b981;
        }

        .browse-btn {
            padding: 12px 16px;
            background: #2a2a3e;
            border: 2px solid #3a3a4e;
            color: #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 16px;
        }
        
        .browse-btn:hover {
            border-color: #007acc;
            background: #3a3a4e;
        }

        .file-info {
            display: none;
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .file-info.active {
            display: block;
        }
        
        .file-name {
            color: #007acc;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .file-size {
            color: #a0a0a0;
            font-size: 13px;
        }
        
        .version-note {
            text-align: center;
            color: #7a7a7a;
            font-size: 12px;
            margin-top: 16px;
            letter-spacing: 0.3px;
        }
    </style>
</head>
<body>
    <div class="top-controls">
        <button class="feedback-btn" onclick="window.location.href='mailto:yunyang.li@yale.edu?subject=Video Annotation Tool Feedback'">
            📧 Feedback
        </button>
        <button class="restart-btn" id="restartBtn" onclick="restartServer()">
            🔄 Restart Server
        </button>
        <button class="update-btn" id="updateBtn" onclick="updateSoftware()">
            ⬇️ Update Software
        </button>
    </div>
    <div class="upload-container">
        <h1>📹 Video Annotation Editor</h1>
        <p class="subtitle">Upload a video or provide a path to begin timeline annotation</p>
        
        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">🎬</div>
            <div class="upload-text">Click to upload or drag & drop</div>
            <div class="upload-hint">Supports MP4, AVI, MOV (max 2GB)</div>
        </div>
        
        <input type="file" id="fileInput" accept="video/*">
        
        <div class="file-info" id="fileInfo">
            <div class="file-name" id="fileName"></div>
            <div class="file-size" id="fileSize"></div>
        </div>
        
        <div class="divider"><span>OR</span></div>
        
        <div class="path-input-group">
            <label>Video File Path</label>
            <input type="text" id="videoPath" placeholder="/path/to/video.mp4">
        </div>
        
        <button id="processBtn" onclick="processVideo()">Start Annotation</button>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-text" id="progressText">Processing video...</div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="progress-percentage" id="progressPercentage">0%</div>
        </div>
        
        <div id="statusMessage"></div>
        <div class="version-note">Version: {{ git_version }}</div>
    </div>
    
    <script>
        async function restartServer() {
            const btn = document.getElementById('restartBtn');
            if (btn) btn.disabled = true;
            const originalText = btn.textContent;
            btn.textContent = 'Restarting...';
            
            showNotification('Restarting server...');
            try {
                const response = await fetch('/admin/restart', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    showNotification('Server restarting. Reloading page...');
                    setTimeout(() => window.location.reload(), 2000);
                } else {
                    const errorMsg = data.error || 'Unknown error';
                    showNotification('Restart failed: ' + errorMsg);
                    btn.textContent = originalText;
                    if (btn) btn.disabled = false;
                }
            } catch (err) {
                showNotification('Restart error: ' + err.message);
                btn.textContent = originalText;
                if (btn) btn.disabled = false;
            }
        }

        async function updateSoftware() {
            const btn = document.getElementById('updateBtn');
            if (btn) btn.disabled = true;
            const originalText = btn.textContent;
            btn.textContent = 'Updating...';
            
            showNotification('Updating software from git...');
            try {
                const response = await fetch('/admin/update', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    showNotification('Update complete. Reloading...');
                    console.log('Update output:', data.output);
                    setTimeout(() => window.location.reload(), 2000);
                } else {
                    const errorMsg = data.error || 'Unknown error';
                    showNotification('Update failed: ' + errorMsg);
                    console.error('Update failed:', data);
                    btn.textContent = originalText;
                    if (btn) btn.disabled = false;
                }
            } catch (err) {
                showNotification('Update error: ' + err.message);
                console.error('Update error:', err);
                btn.textContent = originalText;
                if (btn) btn.disabled = false;
            }
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

        let selectedFile = null;
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const videoPath = document.getElementById('videoPath');
        const processBtn = document.getElementById('processBtn');
        
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(file);
            } else {
                showStatus('Please drop a video file', 'error');
            }
        });
        
        function handleFileSelect(file) {
            selectedFile = file;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = formatFileSize(file.size);
            fileInfo.classList.add('active');
            videoPath.value = '';
        }
        
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
            else if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
            else return (bytes / 1073741824).toFixed(1) + ' GB';
        }
        
        async function processVideo() {
            const path = videoPath.value.trim();
            
            if (!selectedFile && !path) {
                showStatus('Please upload a video file or provide a path', 'error');
                return;
            }
            
            processBtn.disabled = true;
            document.getElementById('progressContainer').classList.add('active');
            
            try {
                if (selectedFile) {
                    await uploadFile(selectedFile);
                } else {
                    await processPath(path);
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
                processBtn.disabled = false;
            }
        }
        
        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('video', file);
            
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 50;
                    updateProgress(percentComplete, 'Uploading video...');
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        pollProgress();
                    } else {
                        showStatus('Error: ' + response.error, 'error');
                        processBtn.disabled = false;
                    }
                } else {
                    showStatus('Upload failed', 'error');
                    processBtn.disabled = false;
                }
            });
            
            xhr.addEventListener('error', () => {
                showStatus('Upload failed', 'error');
                processBtn.disabled = false;
            });
            
            xhr.open('POST', '/upload_video');
            xhr.send(formData);
        }
        
        async function processPath(path) {
            updateProgress(10, 'Processing video path...');
            
            const response = await fetch('/process_path', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path})
            });
            
            const data = await response.json();
            
            if (data.success) {
                pollProgress();
            } else {
                showStatus('Error: ' + data.error, 'error');
                processBtn.disabled = false;
            }
        }
        
        async function pollProgress() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch('/processing_status');
                    const data = await response.json();
                    
                    if (data.status === 'processing') {
                        updateProgress(50 + data.progress * 0.5, data.message);
                    } else if (data.status === 'complete') {
                        clearInterval(interval);
                        updateProgress(100, 'Complete! Redirecting...');
                        showStatus('Video ready! Redirecting to editor...', 'success');
                        setTimeout(() => {
                            window.location.href = '/editor';
                        }, 1500);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        showStatus('Error: ' + data.message, 'error');
                        processBtn.disabled = false;
                    }
                } catch (error) {
                    clearInterval(interval);
                    showStatus('Error checking status', 'error');
                    processBtn.disabled = false;
                }
            }, 500);
        }
        
        function updateProgress(percent, text) {
            document.getElementById('progressBar').style.width = percent + '%';
            document.getElementById('progressPercentage').textContent = Math.round(percent) + '%';
            document.getElementById('progressText').textContent = text;
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = 'status-message status-' + type;
            setTimeout(() => {
                statusDiv.textContent = '';
                statusDiv.className = 'status-message';
            }, 5000);
        }
    </script>
</body>
</html>
"""

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
            width: 100%;
            height: 100%;
            object-fit: contain; /* Scale to fit portrait/landscape while preserving aspect */
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
            overflow-y: auto; /* Allow vertical scrolling when many tracks */
        }
        
        .timeline-content {
            position: relative;
            height: auto;      /* Grow with added tracks */
            min-height: 100%;  /* But never smaller than viewport */
            min-width: 100%;
            padding-bottom: 20px; /* Breathing room for bottom track */
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
            overflow: visible; /* Allow selection glow to show at edges (time 0) */
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
            cursor: grab; /* Clear affordance for drag */
            user-select: none;
            display: flex;
            align-items: center;
            padding: 0 12px; /* Larger grab area */
            overflow: hidden;
            transition: box-shadow 0.2s;
            z-index: 5; /* Base z-index for segments */
        }
        
        .timeline-segment:hover {
            box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.3);
            z-index: 10;
        }
        
        .timeline-segment.selected {
            border-color: #ffeb3b;
            box-shadow: 
                0 0 0 3px rgba(255, 235, 59, 0.8),
                0 0 12px rgba(255, 235, 59, 0.6),
                inset 0 0 0 2px rgba(255, 235, 59, 0.9); /* Inner outline still visible if outer is clipped */
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
            width: 10px; /* Easier to grab on trackpads */
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
            position: relative;
            z-index: 1;
            margin-bottom: 12px; /* Add breathing room between stacked fields */
        }
        
        .property-input:focus {
            outline: none;
            border-color: #007acc;
            z-index: 100;
        }
        
        /* Ensure dropdown menus appear above neighboring fields */
        .annotation-dialog select.property-input {
            position: relative;
            z-index: 5;
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
            position: absolute;
            top: 50px;
            right: 20px;
            transform: none;
            background: #2d2d30;
            border: 1px solid #464647;
            border-radius: 8px;
            padding: 16px;
            width: 260px;
            z-index: 1000;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
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
        <button class="toolbar-btn" onclick="loadNewVideo()" title="Load a different video">🔄 New Video</button>
        <div style="width: 1px; height: 20px; background: #464647;"></div>
        <button class="toolbar-btn" onclick="showNewAnnotationDialog()" style="background: #007acc; color: white;">➕ Add Annotation</button>
        <button class="toolbar-btn" onclick="duplicateSelected()">📋 Duplicate</button>
        <button class="toolbar-btn" onclick="deleteSelected()">🗑️ Delete</button>
        <div style="width: 1px; height: 20px; background: #464647; margin-left: auto;"></div>
        <button class="toolbar-btn" onclick="saveProject()">💾 Save</button>
        <button class="toolbar-btn" onclick="loadProject()">📁 Open</button>
        <button class="toolbar-btn" id="shortcutsBtn" onclick="showShortcuts()">⌨️ Shortcuts</button>
    </div>
    <!-- Hidden file input for loading projects -->
    <input type="file" id="projectFileInput" accept=".json,application/json" style="display: none;">
    
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
            <button class="toolbar-btn" onclick="addTrack()">Add Track</button>
            <button class="toolbar-btn" onclick="removeTrack()" id="removeTrackBtn" disabled>Remove Track</button>
            <span style="font-size: 11px; color: #969696;">Tracks: <span id="trackCountLabel">1</span></span>
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
        <div class="context-menu-separator"></div>
        <div class="context-menu-item" onclick="setStartToCurrentTime(selectedSegment?.id)">Set Start @ Current</div>
        <div class="context-menu-item" onclick="setEndToCurrentTime(selectedSegment?.id)">Set End @ Current</div>
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
            <span>Copy / Paste</span>
            <span class="shortcut-key">Cmd/Ctrl + C / Cmd/Ctrl + V</span>
        </div>
        <div class="shortcut-item">
            <span>Set Start @ Current</span>
            <span class="shortcut-key">[</span>
        </div>
        <div class="shortcut-item">
            <span>Set End @ Current</span>
            <span class="shortcut-key">]</span>
        </div>
        <div class="shortcut-item">
            <span>Save Project</span>
            <span class="shortcut-key">Cmd/Ctrl + S</span>
        </div>
        <button class="toolbar-btn" onclick="hideShortcuts()" style="margin-top: 15px; width: 100%;">Close</button>
    </div>
    
    <!-- Save Dialog -->
    <div class="shortcuts-modal" id="saveDialog" style="display: none; max-width: 500px;">
        <h3>💾 Save Annotations</h3>
        <div style="margin: 20px 0;">
            <label style="display: block; color: #969696; margin-bottom: 8px; font-size: 12px;">Project Name</label>
            <input type="text" id="projectName" class="property-input" placeholder="Enter project name (optional)" 
                   style="width: 100%; background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 10px; font-size: 14px;">
            <div style="margin-top: 10px; font-size: 11px; color: #969696;">
                Leave empty for auto-generated name with timestamp
            </div>
            <div style="margin-top: 15px; padding: 10px; background: rgba(0, 122, 204, 0.1); border-left: 3px solid #007acc; border-radius: 4px;">
                <div style="font-size: 11px; color: #cccccc;">
                    📁 <strong>Your browser will prompt you to choose where to save the file</strong>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 10px;">
            <button class="toolbar-btn" onclick="confirmSave()" style="flex: 1; background: #007acc; color: white;">Save</button>
            <button class="toolbar-btn" onclick="hideSaveDialog()" style="flex: 1;">Cancel</button>
        </div>
    </div>
    
    <!-- New Annotation Dialog -->
    <div class="annotation-dialog" id="newAnnotationDialog">
        <h3>➕ Add New Annotation</h3>
        <div style="margin: 20px 0;">
            <label style="display: block; color: #969696; margin-bottom: 5px;">Task Category</label>
            <select id="newTaskCategory" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
                <option value="MDS-UPDRS">MDS-UPDRS</option>
                <option value="MoCA">MoCA</option>
            </select>
            
            <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Task Type</label>
            <select id="newTaskType" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
                <option value="3.1 Speech">3.1 Speech</option>
                <option value="3.2 Facial Expression">3.2 Facial Expression</option>
                <option value="3.4 Finger Tapping (Left)">3.4 Finger Tapping (L)</option>
                <option value="3.4 Finger Tapping (Right)">3.4 Finger Tapping (R)</option>
                <option value="3.5 Hand Movements (Left)">3.5 Hand Movements (L)</option>
                <option value="3.5 Hand Movements (Right)">3.5 Hand Movements (R)</option>
                <option value="3.6 Pronation-Supination Movements of Hands (Left)">3.6 Pronation-Supination (L)</option>
                <option value="3.6 Pronation-Supination Movements of Hands (Right)">3.6 Pronation-Supination (R)</option>
                <option value="3.7 Toe Tapping (Left)">3.7 Toe Tapping (L)</option>
                <option value="3.7 Toe Tapping (Right)">3.7 Toe Tapping (R)</option>
                <option value="3.8 Leg Agility (Left)">3.8 Leg Agility (L)</option>
                <option value="3.8 Leg Agility (Right)">3.8 Leg Agility (R)</option>
                <option value="3.9 Arising from Chair">3.9 Arising from Chair</option>
                <option value="3.12 Postural Stability">3.12 Postural Stability</option>
                <option value="3.13 Posture">3.13 Posture</option>
                <option value="3.14 Global Spontaneity of Movement (Body Bradykinesia)">3.14 Global Spontaneity of Movement (Body Bradykinesia)</option>
                <option value="3.15 Postural Tremor of the Hands">3.15 Postural Tremor of the Hands</option>
                <option value="3.16 Kinetic Tremor of the Hands (Left)">3.16 Kinetic Tremor of the Hands (L)</option>
                <option value="3.16 Kinetic Tremor of the Hands (Right)">3.16 Kinetic Tremor of the Hands (R)</option>
                <option value="3.17 Rest Tremor">3.17 Rest Tremor</option>
                <option value="4.1 Dyskinesias (Yes/No)">4.1 Dyskinesias (Yes/No)</option>
                <!-- TUG-related (forward-compatible) -->
                <option value="3.10 Gait">3.10 Gait</option>
                <option value="3.11 Freezing of Gait">3.11 Freezing of Gait</option>
                <option value="Turning">Turning</option>
                <!-- MoCA domains -->
                <option value="Orientation">Orientation (MoCA)</option>
            <option value="Visuospatial-Executive">Visuospatial-Executive (MoCA)</option>
            <option value="Naming">Naming (MoCA)</option>
            <option value="Memory">Memory (MoCA)</option>
            <option value="Attention">Attention (MoCA)</option>
            <option value="Language">Language (MoCA)</option>
            <option value="Abstraction">Abstraction (MoCA)</option>
            <option value="Delayed Recall">Delayed Recall (MoCA)</option>
        </select>
        <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Duration (seconds)</label>
        <input type="number" id="newDuration" value="5" min="0.5" max="30" step="0.5" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
        
        <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Severity (UPDRS 0-4)</label>
        <input type="number" id="newSeverity" value="0" min="0" max="4" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;">
            
            <label style="display: block; color: #969696; margin-bottom: 5px; margin-top: 15px;">Track</label>
            <select id="newTrack" class="property-input" style="background: #3c3c3c; border: 1px solid #464647; color: #cccccc; padding: 8px; width: 100%;"></select>
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
        let justFinishedDraggingSegment = false; // Ignore stray clicks after segment drag/resize
        let dragStartX = 0;
        let segmentStartPos = 0;
        let segmentStartWidth = 0;
        let timelineZoom = 10; // pixels per second
        let videoDuration = 0;
        let annotations = [];
        let clipboard = null;
        let trackCount = 1; // Number of visible tracks
        
        // Store last used settings for convenience (but always allow changing)
        let lastUsedSettings = {
            task: '3.1 Speech', // Default to a real option so first annotation has a label
            category: 'MDS-UPDRS',
            duration: 5,
            severity: 0,
            track: 0
        };
        
        // Deterministic color palette for task-based coloring
        const COLOR_PALETTE = [
            '#4c9aff', '#69f0ae', '#ffab40', '#ff5252',
            '#b388ff', '#4dd0e1', '#ff80ab', '#90a4ae',
            '#ffd54f', '#64b5f6', '#ce93d8', '#a5d6a7'
        ];
        
        function hashString(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                hash = ((hash << 5) - hash) + str.charCodeAt(i);
                hash |= 0; // Convert to 32bit integer
            }
            return Math.abs(hash);
        }
        
        function getColorForTask(task) {
            const key = task || 'default';
            const idx = hashString(key) % COLOR_PALETTE.length;
            return COLOR_PALETTE[idx];
        }
        
        function shadeColor(color, percent) {
            const f = parseInt(color.slice(1), 16);
            const t = percent < 0 ? 0 : 255;
            const p = Math.abs(percent) / 100;
            const R = f >> 16;
            const G = f >> 8 & 0x00FF;
            const B = f & 0x0000FF;
            const newColor = '#' + (
                0x1000000 +
                (Math.round((t - R) * p) + R) * 0x10000 +
                (Math.round((t - G) * p) + G) * 0x100 +
                (Math.round((t - B) * p) + B)
            ).toString(16).slice(1);
            return newColor;
        }
        
        // Task catalog grouped by category to keep dropdowns manageable
        const TASK_OPTIONS = {
            'MDS-UPDRS': [
                { value: '3.1 Speech', label: '3.1 Speech' },
                { value: '3.2 Facial Expression', label: '3.2 Facial Expression' },
                { value: '3.4 Finger Tapping (Left)', label: '3.4 Finger Tapping (L)' },
                { value: '3.4 Finger Tapping (Right)', label: '3.4 Finger Tapping (R)' },
                { value: '3.5 Hand Movements (Left)', label: '3.5 Hand Movements (L)' },
                { value: '3.5 Hand Movements (Right)', label: '3.5 Hand Movements (R)' },
                { value: '3.6 Pronation-Supination Movements of Hands (Left)', label: '3.6 Pronation-Supination (L)' },
                { value: '3.6 Pronation-Supination Movements of Hands (Right)', label: '3.6 Pronation-Supination (R)' },
                { value: '3.7 Toe Tapping (Left)', label: '3.7 Toe Tapping (L)' },
                { value: '3.7 Toe Tapping (Right)', label: '3.7 Toe Tapping (R)' },
                { value: '3.8 Leg Agility (Left)', label: '3.8 Leg Agility (L)' },
                { value: '3.8 Leg Agility (Right)', label: '3.8 Leg Agility (R)' },
                { value: '3.9 Arising from Chair', label: '3.9 Arising from Chair' },
                { value: '3.10 Gait', label: '3.10 Gait' },
                { value: '3.11 Freezing of Gait', label: '3.11 Freezing of Gait' },
                { value: '3.12 Postural Stability', label: '3.12 Postural Stability' },
                { value: '3.13 Posture', label: '3.13 Posture' },
                { value: '3.14 Global Spontaneity of Movement (Body Bradykinesia)', label: '3.14 Global Spontaneity of Movement (Body Bradykinesia)' },
                { value: '3.15 Postural Tremor of the Hands', label: '3.15 Postural Tremor of the Hands' },
                { value: '3.16 Kinetic Tremor of the Hands (Left)', label: '3.16 Kinetic Tremor of the Hands (L)' },
                { value: '3.16 Kinetic Tremor of the Hands (Right)', label: '3.16 Kinetic Tremor of the Hands (R)' },
                { value: '3.17 Rest Tremor', label: '3.17 Rest Tremor' },
                { value: '4.1 Dyskinesias (Yes/No)', label: '4.1 Dyskinesias (Yes/No)' },
                { value: 'Turning', label: 'Turning' }
            ],
            'MoCA': [
                { value: 'Orientation', label: 'Orientation (MoCA)' },
                { value: 'Visuospatial-Executive', label: 'Visuospatial-Executive (MoCA)' },
                { value: 'Naming', label: 'Naming (MoCA)' },
                { value: 'Memory', label: 'Memory (MoCA)' },
                { value: 'Attention', label: 'Attention (MoCA)' },
                { value: 'Language', label: 'Language (MoCA)' },
                { value: 'Abstraction', label: 'Abstraction (MoCA)' },
                { value: 'Delayed Recall', label: 'Delayed Recall (MoCA)' }
            ]
        };
        
        function getCategoryForTask(task) {
            if (TASK_OPTIONS['MoCA'].some(opt => opt.value === task)) {
                return 'MoCA';
            }
            return 'MDS-UPDRS';
        }
        
        function buildCategoryOptionsHTML(selectedCategory) {
            return Object.keys(TASK_OPTIONS)
                .map(cat => `<option value="${cat}" ${selectedCategory === cat ? 'selected' : ''}>${cat}</option>`)
                .join('');
        }
        
        function buildTaskOptionsHTML(category, selectedTask) {
            const tasks = TASK_OPTIONS[category] || [];
            return tasks.map(opt => `<option value="${opt.value}" ${opt.value === selectedTask ? 'selected' : ''}>${opt.label}</option>`).join('');
        }
        
        function populateTaskSelect(selectEl, category, selectedTask) {
            const tasks = TASK_OPTIONS[category] || [];
            selectEl.innerHTML = buildTaskOptionsHTML(category, selectedTask || tasks[0]?.value || '');
            if (selectedTask && !tasks.some(opt => opt.value === selectedTask) && tasks.length > 0) {
                selectEl.value = tasks[0].value;
            }
        }
        
        function updateNewTaskOptions(selectedTask) {
            const category = document.getElementById('newTaskCategory').value;
            populateTaskSelect(document.getElementById('newTaskType'), category, selectedTask);
        }
        
        const taskCategorySelect = document.getElementById('newTaskCategory');
        if (taskCategorySelect) {
            taskCategorySelect.addEventListener('change', () => updateNewTaskOptions());
        }
        updateNewTaskOptions(lastUsedSettings.task);

        function buildTrackOptionsHTML(selectedTrack = 0) {
            let options = '';
            for (let i = 0; i < trackCount; i++) {
                options += `<option value="${i}" ${selectedTrack === i ? 'selected' : ''}>Track ${i + 1}</option>`;
            }
            return options;
        }
        
        function rebuildTracks() {
            const tracksContainer = document.getElementById('timelineTracks');
            if (!tracksContainer) return;
            tracksContainer.innerHTML = '';
            for (let i = 0; i < trackCount; i++) {
                const track = document.createElement('div');
                track.className = 'timeline-track';
                track.dataset.track = i.toString();
                track.style.height = '60px';
                
                const label = document.createElement('div');
                label.className = 'track-label';
                label.textContent = `Track ${i + 1}`;
                track.appendChild(label);
                
                tracksContainer.appendChild(track);
            }
        }
        
        function refreshTrackSelectors(preferredTrack) {
            const newTrackSelect = document.getElementById('newTrack');
            if (newTrackSelect) {
                const nextValue = preferredTrack !== undefined ? preferredTrack : parseInt(newTrackSelect.value || '0');
                const safeValue = Math.max(0, Math.min(nextValue, trackCount - 1));
                newTrackSelect.innerHTML = buildTrackOptionsHTML(safeValue);
                newTrackSelect.value = safeValue.toString();
            }
            
            if (selectedSegment) {
                updateProperties(selectedSegment);
            }
            
            const removeTrackBtn = document.getElementById('removeTrackBtn');
            if (removeTrackBtn) {
                removeTrackBtn.disabled = trackCount <= 1;
            }
            
            const trackCountLabel = document.getElementById('trackCountLabel');
            if (trackCountLabel) {
                trackCountLabel.textContent = trackCount;
            }
        }
        
        function setTrackCount(count) {
            trackCount = Math.max(1, count);
            rebuildTracks();
            refreshTrackSelectors();
            updateTimeline();
            if (selectedSegment) {
                selectSegment(selectedSegment.id);
            }
        }
        
        function addTrack() {
            setTrackCount(trackCount + 1);
        }
        
        function removeTrack() {
            if (trackCount <= 1) return;
            trackCount -= 1;
            annotations.forEach(ann => {
                if (ann.track >= trackCount) {
                    ann.track = trackCount - 1;
                }
            });
            setTrackCount(trackCount);
        }
        
        // Initialize tracks and selectors
        setTrackCount(trackCount);
        
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
                if (isPlayheadDragging || justFinishedDraggingPlayhead || isDragging || isResizing || justFinishedDraggingSegment) {
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
            updatePlayhead();
        }
        
        function updatePlayhead() {
            const playhead = document.getElementById('playhead');
            const position = video.currentTime * timelineZoom + 130;
            playhead.style.left = position + 'px';
        }
        
        function applyZoomPreservingView(previousZoom) {
            const scrollable = document.getElementById('timelineScrollable');
            if (!scrollable || !videoDuration || videoDuration <= 0) {
                updateTimeline();
                return;
            }
            
            // Keep the playhead anchored relative to current viewport position
            const playheadBefore = video.currentTime * previousZoom + 130;
            const offsetFromLeft = playheadBefore - scrollable.scrollLeft;
            
            updateTimeline();
            
            const playheadAfter = video.currentTime * timelineZoom + 130;
            const targetScroll = playheadAfter - offsetFromLeft;
            const maxScroll = Math.max(0, scrollable.scrollWidth - scrollable.clientWidth);
            scrollable.scrollLeft = Math.max(0, Math.min(targetScroll, maxScroll));
        }
        
        function zoomIn() {
            const previousZoom = timelineZoom;
            timelineZoom = Math.min(100, timelineZoom * 1.5);
            applyZoomPreservingView(previousZoom);
        }
        
        function zoomOut() {
            const previousZoom = timelineZoom;
            timelineZoom = Math.max(2, timelineZoom / 1.5);
            applyZoomPreservingView(previousZoom);
        }
        
        function fitToWindow() {
            const scrollable = document.getElementById('timelineScrollable');
            const previousZoom = timelineZoom;
            const availableWidth = scrollable.clientWidth - 130;
            timelineZoom = availableWidth / videoDuration;
            applyZoomPreservingView(previousZoom);
        }
        
        // Segment management
        function showNewAnnotationDialog() {
            // Use last settings for convenience, but ensure form is editable
            document.getElementById('newTaskCategory').value = lastUsedSettings.category;
            updateNewTaskOptions(lastUsedSettings.task);
            const taskSelect = document.getElementById('newTaskType');
            // If the stored task isn't available, fall back to the first option
            if (!taskSelect.value) {
                const categoryOptions = TASK_OPTIONS[lastUsedSettings.category] || [];
                taskSelect.value = categoryOptions[0]?.value || '';
            }
            document.getElementById('newDuration').value = lastUsedSettings.duration.toString();
            document.getElementById('newSeverity').value = lastUsedSettings.severity.toString();
            refreshTrackSelectors(lastUsedSettings.track || 0);
            document.getElementById('newTrack').value = Math.min(lastUsedSettings.track || 0, trackCount - 1).toString();
            
            // Ensure the select elements are not disabled and have all options
            taskSelect.disabled = false;
            
            document.getElementById('newAnnotationDialog').style.display = 'block';
        }
        
        function hideNewAnnotationDialog() {
            document.getElementById('newAnnotationDialog').style.display = 'none';
        }
        
        function createNewAnnotation() {
            let task = document.getElementById('newTaskType').value;
            const side = 'n/a'; // Side selection removed; default to not applicable
            const category = document.getElementById('newTaskCategory').value;
            const duration = parseFloat(document.getElementById('newDuration').value);
            const severity = parseInt(document.getElementById('newSeverity').value);
            const rawTrack = parseInt(document.getElementById('newTrack').value);
            const track = Number.isNaN(rawTrack) ? 0 : Math.min(rawTrack, trackCount - 1);
            
            if (!task) {
                const tasks = TASK_OPTIONS[category] || [];
                task = tasks[0]?.value || 'Untitled';
            }
            
            // Save settings for next time (convenience feature)
            lastUsedSettings = {
                task: task,
                category: category,
                duration: duration,
                severity: severity,
                track: track
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
                category: category,
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
            
            // Color coding by task
            const baseColor = getColorForTask(annotation.task);
            const darker = shadeColor(baseColor, -15);
            segment.style.background = `linear-gradient(135deg, ${baseColor}, ${darker})`;
            segment.style.borderColor = darker;
            
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
                // Select immediately on mousedown for instant feedback
                selectSegment(annotation.id);
                
                if (e.target.classList.contains('segment-resize-handle')) {
                    startResize(e, annotation, e.target.classList.contains('left'));
                } else {
                    startDrag(e, annotation);
                }
            });
            
            segment.addEventListener('click', (e) => {
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
            let pendingFrame = null;
            let pendingLeft = segmentStartPos * timelineZoom;
            e.preventDefault();
            
            const handleDrag = (e) => {
                const deltaX = e.clientX - dragStartX;
                const deltaY = e.clientY - startY;
                
                // On mac trackpads small jitters can prevent the drag; accept slight movement
                const dragStartSatisfied = hasStartedDragging || Math.abs(e.clientX - startX) >= 2;
                
                // Only start dragging if moved beyond threshold
                if (!hasStartedDragging && (!dragStartSatisfied || Math.abs(e.clientX - startX) < dragThreshold)) {
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
                pendingLeft = annotation.start * timelineZoom;
                if (!pendingFrame) {
                    pendingFrame = requestAnimationFrame(() => {
                        segment.style.left = pendingLeft + 'px';
                        pendingFrame = null;
                    });
                }
                
                // Vertical movement - track switching based on current pointer position
                const tracks = Array.from(document.querySelectorAll('.timeline-track'));
                const pointerY = e.clientY;
                const targetTrackEl = tracks.find(trackEl => {
                    const rect = trackEl.getBoundingClientRect();
                    return pointerY >= rect.top && pointerY <= rect.bottom;
                });
                if (targetTrackEl) {
                    const newTrack = parseInt(targetTrackEl.dataset.track);
                    if (!Number.isNaN(newTrack) && newTrack !== annotation.track) {
                        annotation.track = newTrack;
                        targetTrackEl.appendChild(segment);
                        // Keep the segment marked as selected without full re-render
                        segment.classList.add('selected');
                        selectedSegment = annotation;
                        updateProperties(annotation);
                    }
                }
            };
            
            const stopDrag = () => {
                if (hasStartedDragging) {
                    isDragging = false;
                    segment.classList.remove('dragging');
                    if (pendingFrame) {
                        cancelAnimationFrame(pendingFrame);
                        pendingFrame = null;
                        segment.style.left = (annotation.start * timelineZoom) + 'px';
                    }
                    updateProperties(annotation);
                    justFinishedDraggingSegment = true;
                    setTimeout(() => { justFinishedDraggingSegment = false; }, 120);
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
            e.preventDefault();
            
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
                justFinishedDraggingSegment = true;
                setTimeout(() => { justFinishedDraggingSegment = false; }, 120);
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
            const category = annotation.category || getCategoryForTask(annotation.task);
            content.innerHTML = `
                <div class="property-group">
                    <div class="property-label">Timing</div>
                    <input type="number" class="property-input" value="${annotation.start.toFixed(2)}" 
                           onchange="updateAnnotation(${annotation.id}, 'start', parseFloat(this.value))"
                           step="0.1" min="0" max="${videoDuration}">
                    <input type="number" class="property-input" value="${annotation.end.toFixed(2)}" 
                           onchange="updateAnnotation(${annotation.id}, 'end', parseFloat(this.value))"
                           step="0.1" min="0" max="${videoDuration}" style="margin-top: 5px;">
                    <div style="display: flex; gap: 8px; margin-top: 6px;">
                        <button class="toolbar-btn" style="flex: 1;" onclick="setStartToCurrentTime(${annotation.id})">Set Start @ Current</button>
                        <button class="toolbar-btn" style="flex: 1;" onclick="setEndToCurrentTime(${annotation.id})">Set End @ Current</button>
                    </div>
                </div>
                
                <div class="property-group">
                    <div class="property-label">Task Category</div>
                    <select class="property-input" onchange="updateAnnotation(${annotation.id}, 'category', this.value)">
                        ${buildCategoryOptionsHTML(category)}
                    </select>
                </div>
                
                <div class="property-group">
                    <div class="property-label">Task Type</div>
                    <select class="property-input" onchange="updateAnnotation(${annotation.id}, 'task', this.value)">
                        ${buildTaskOptionsHTML(category, annotation.task)}
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
                        ${buildTrackOptionsHTML(annotation.track)}
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
                
                if (field === 'category') {
                    const tasks = TASK_OPTIONS[value] || [];
                    if (annotation.task && !tasks.some(opt => opt.value === annotation.task) && tasks.length > 0) {
                        annotation.task = tasks[0].value;
                    }
                }
                
                // Don't override track unless it doesn't exist
                if (annotation.track === undefined) {
                    annotation.track = 0;
                }
                
                if (field === 'track' && annotation.track >= trackCount) {
                    setTrackCount(annotation.track + 1);
                    selectSegment(id);
                    return;
                }
                
                renderSegments();
                selectSegment(id);
            }
        }
        
        function setStartToCurrentTime(id) {
            const annotation = annotations.find(ann => ann.id === id);
            if (!annotation) return;
            const newStart = Math.max(0, Math.min(video.currentTime, annotation.end - 0.5));
            annotation.start = newStart;
            if (annotation.end - annotation.start < 0.5) {
                annotation.end = Math.min(videoDuration, annotation.start + 0.5);
            }
            renderSegments();
            selectSegment(id);
        }
        
        function setEndToCurrentTime(id) {
            const annotation = annotations.find(ann => ann.id === id);
            if (!annotation) return;
            const newEnd = Math.min(videoDuration, Math.max(video.currentTime, annotation.start + 0.5));
            annotation.end = newEnd;
            renderSegments();
            selectSegment(id);
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
            // Measure menu first, then place with clamping
            menu.style.visibility = 'hidden';
            menu.style.display = 'block';
            menu.style.left = '0px';
            menu.style.top = '0px';
            const menuRect = menu.getBoundingClientRect();
            let posX = e.clientX;
            let posY = e.clientY;
            if (posX + menuRect.width > window.innerWidth) {
                posX = Math.max(0, window.innerWidth - menuRect.width - 8);
            }
            if (posY + menuRect.height > window.innerHeight) {
                posY = Math.max(0, window.innerHeight - menuRect.height - 8);
            }
            menu.style.left = posX + 'px';
            menu.style.top = posY + 'px';
            menu.style.visibility = 'visible';
            
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
                if (newSegment.track === undefined) {
                    newSegment.track = 0;
                }
                annotations.push(newSegment);
                if (newSegment.track >= trackCount) {
                    setTrackCount(newSegment.track + 1);
                } else {
                    renderSegments();
                }
                selectSegment(newSegment.id);
            }
        }
        
        // Save/Load
        function saveProject() {
            // Show save dialog
            document.getElementById('saveDialog').style.display = 'block';
            // Pre-fill with video name if available
            const videoName = '{{ original_video_name }}' || 'project';
            const baseName = videoName.replace(/\.[^/.]+$/, ''); // Remove extension
            document.getElementById('projectName').value = baseName;
            document.getElementById('projectName').select(); // Select text for easy editing
        }
        
        function hideSaveDialog() {
            document.getElementById('saveDialog').style.display = 'none';
        }
        
        function confirmSave() {
            const customName = document.getElementById('projectName').value.trim();
            
            // Generate filename
            let filename;
            if (customName) {
                filename = `${customName}_${new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19)}.json`;
            } else {
                const videoName = '{{ original_video_name }}' || 'project';
                const baseName = videoName.replace(/\.[^/.]+$/, '');
                filename = `${baseName}_annotations_${new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19)}.json`;
            }
            
            // Prepare data
            const data = {
                annotations: annotations,
                videoDuration: videoDuration,
                timestamp: new Date().toISOString(),
                video: '{{ original_video_path }}'
            };
            
            // Create blob and download
            const jsonStr = JSON.stringify(data, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            // Create temporary download link
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
            
            showNotification(`Downloading: ${filename}`);
            hideSaveDialog();
        }
        
        function loadNewVideo() {
            // Check if there are unsaved annotations; ask for a single confirmation
            if (annotations.length > 0) {
                const confirmMsg = `You have ${annotations.length} annotation(s). Have you saved them? Starting a new video will discard them. Continue?`;
                if (!confirm(confirmMsg)) {
                    return;
                }
            }
            // Redirect to upload page to start fresh
            window.location.href = '/';
        }
        
        function loadProject() {
            const input = document.getElementById('projectFileInput');
            if (!input) return;
            input.value = '';
            input.click();
        }
        
        const projectFileInput = document.getElementById('projectFileInput');
        if (projectFileInput) {
            projectFileInput.addEventListener('change', (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                
                const reader = new FileReader();
                reader.onload = () => {
                    try {
                        const data = JSON.parse(reader.result || '{}');
                        const loadedAnnotations = (data.annotations || []).map(ann => ({
                            ...ann,
                            track: ann.track === undefined ? 0 : ann.track
                        }));
                        const message = `Loaded ${loadedAnnotations.length} annotation(s).\n\nOK = Append to current\nCancel = Start fresh (replace current)`;
                        const append = confirm(message);
                        if (!append) {
                            annotations = [];
                            selectedSegment = null;
                        }
                        annotations = append ? annotations.concat(loadedAnnotations) : loadedAnnotations;
                        
                        // Update duration if provided; otherwise keep current
                        if (data.videoDuration && !Number.isNaN(data.videoDuration)) {
                            videoDuration = data.videoDuration;
                        }
                        
                        // Ensure track count covers new annotations
                        const maxTrack = annotations.reduce((max, ann) => Math.max(max, ann.track || 0), 0);
                        setTrackCount(Math.max(trackCount, maxTrack + 1));
                        
                        // Re-render
                        renderSegments();
                        deselectAll();
                        updateTimeline();
                        showNotification(`Loaded ${loadedAnnotations.length} annotation(s)${append ? ' (appended)' : ''}.`);
                    } catch (err) {
                        alert('Failed to load project: ' + err.message);
                    } finally {
                        projectFileInput.value = '';
                    }
                };
                reader.readAsText(file);
            });
        }
        
        async function updateSoftware() {
            const btn = document.getElementById('updateBtn');
            if (btn) btn.disabled = true;
            showNotification('Updating software from git...');
            try {
                const response = await fetch('/admin/update', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    showNotification('Update complete. Reload the page to use the latest code.');
                    console.log('Update output:', data.output);
                } else {
                    const errorMsg = data.error || 'Unknown error';
                    showNotification('Update failed: ' + errorMsg);
                    console.error('Update failed:', data);
                }
            } catch (err) {
                showNotification('Update error: ' + err.message);
                console.error('Update error:', err);
            } finally {
                if (btn) btn.disabled = false;
            }
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
            const modal = document.getElementById('shortcutsModal');
            const btn = document.getElementById('shortcutsBtn');
            modal.style.display = 'block';
            // Position near the button like a dropdown
            if (btn) {
                const rect = btn.getBoundingClientRect();
                const modalRect = modal.getBoundingClientRect();
                const scrollTop = window.scrollY || document.documentElement.scrollTop;
                const scrollLeft = window.scrollX || document.documentElement.scrollLeft;
                let left = rect.left + scrollLeft;
                const top = rect.bottom + scrollTop + 6;
                if (left + modalRect.width > window.innerWidth - 12) {
                    left = window.innerWidth - modalRect.width - 12;
                }
                modal.style.top = `${top}px`;
                modal.style.left = `${left}px`;
            }
            setTimeout(() => document.addEventListener('click', hideShortcutsOnOutside), 0);
        }
        
        function hideShortcuts() {
            document.getElementById('shortcutsModal').style.display = 'none';
            document.removeEventListener('click', hideShortcutsOnOutside);
        }
        
        function hideShortcutsOnOutside(e) {
            const modal = document.getElementById('shortcutsModal');
            const btn = document.getElementById('shortcutsBtn');
            if (!modal || !btn) return;
            if (!modal.contains(e.target) && e.target !== btn) {
                hideShortcuts();
            }
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
                case 'c':
                    if (e.metaKey || e.ctrlKey) {
                        e.preventDefault();
                        copySegment();
                    }
                    break;
                case 'v':
                    if (e.metaKey || e.ctrlKey) {
                        e.preventDefault();
                        pasteSegment();
                    }
                    break;
                case '[':
                    if (selectedSegment) {
                        e.preventDefault();
                        setStartToCurrentTime(selectedSegment.id);
                    }
                    break;
                case ']':
                    if (selectedSegment) {
                        e.preventDefault();
                        setEndToCurrentTime(selectedSegment.id);
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
                annotations = (data.annotations || []).map(ann => ({
                    ...ann,
                    track: ann.track === undefined ? 0 : ann.track
                }));
                const maxTrack = annotations.reduce((max, ann) => Math.max(max, ann.track || 0), 0);
                setTrackCount(Math.max(trackCount, maxTrack + 1));
                renderSegments();
            });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Landing page with upload interface"""
    return render_template_string(UPLOAD_TEMPLATE, git_version=GIT_VERSION)

@app.route('/editor')
def editor():
    """Editor interface (only accessible after video is processed)"""
    if not video_ready or not current_video_path:
        return redirect(url_for('index'))
    # Bust browser cache by tagging video URL with version
    video_url = url_for('serve_video', v=video_version)
    return render_template_string(HTML_TEMPLATE, video_url=video_url, original_video_name=original_video_name or '')

@app.route('/serve_video')
def serve_video():
    if video_ready and current_video_path and os.path.exists(current_video_path):
        response = make_response(send_file(current_video_path, mimetype='video/mp4'))
        # Prevent caching to ensure fresh loads when swapping videos
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return '', 404

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    global processing_status
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = file.filename
        upload_path = os.path.join(upload_folder, filename)
        file.save(upload_path)
        
        # Start processing in background thread
        import threading
        reset_video_state()
        processing_status = {"status": "processing", "progress": 0, "message": "Starting conversion..."}
        thread = threading.Thread(target=process_video_file, args=(upload_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process_path', methods=['POST'])
def process_path():
    """Handle video path input"""
    global processing_status
    
    try:
        data = request.json
        video_path = data.get('path', '').strip()
        
        if not video_path:
            return jsonify({'success': False, 'error': 'No path provided'}), 400
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'File not found: {video_path}'}), 404
        
        # Start processing in background thread
        import threading
        reset_video_state()
        processing_status = {"status": "processing", "progress": 0, "message": "Starting conversion..."}
        thread = threading.Thread(target=process_video_file, args=(video_path,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/processing_status')
def get_processing_status():
    """Get current processing status"""
    return jsonify(processing_status)

@app.route('/get_annotations')
def get_annotations():
    return jsonify({'annotations': annotations})


@app.route('/admin/restart', methods=['POST'])
def restart_server_route():
    """Restart the server without git pull."""
    try:
        def restart_server():
            """Restart the server after a short delay."""
            time.sleep(1)
            print("Restarting server...")
            
            # Close all open file descriptors except stdin, stdout, stderr
            try:
                import psutil
                p = psutil.Process(os.getpid())
                for handler in p.open_files() + p.connections():
                    try:
                        os.close(handler.fd)
                    except Exception:
                        pass
            except ImportError:
                # Fallback if psutil is not installed
                import resource
                maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
                if (maxfd == resource.RLIM_INFINITY):
                    maxfd = 1024
                for fd in range(3, maxfd):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            except Exception as e:
                print(f"Error closing FDs: {e}")

            os.execv(sys.executable, [sys.executable] + sys.argv)

        # Start restart in background
        threading.Thread(target=restart_server).start()

        return jsonify({
            'success': True,
            'output': "Server is restarting..."
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/admin/update', methods=['POST'])
def update_software():
    """Run git pull to update the codebase."""
    try:
        result = subprocess.run(
            ['git', '-C', REPO_ROOT, 'pull'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return jsonify({
                'success': False,
                'output': result.stdout.strip(),
                'error': result.stderr.strip()
            }), 500

        def restart_server():
            """Restart the server after a short delay."""
            time.sleep(1)
            print("Restarting server...")
            
            # Close all open file descriptors except stdin, stdout, stderr
            try:
                import psutil
                p = psutil.Process(os.getpid())
                for handler in p.open_files() + p.connections():
                    try:
                        os.close(handler.fd)
                    except Exception:
                        pass
            except ImportError:
                # Fallback if psutil is not installed
                import resource
                maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
                if (maxfd == resource.RLIM_INFINITY):
                    maxfd = 1024
                for fd in range(3, maxfd):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            except Exception as e:
                print(f"Error closing FDs: {e}")

            os.execv(sys.executable, [sys.executable] + sys.argv)

        # Start restart in background
        threading.Thread(target=restart_server).start()

        return jsonify({
            'success': True,
            'output': result.stdout.strip() + "\nServer is restarting..."
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def convert_to_720p(input_path, output_dir=None, progress_callback=None, output_path=None):
    """
    Convert video to 720p resolution for better performance
    
    Args:
        input_path: Path to input video
        output_dir: Directory for output video (None = create temp dir)
        progress_callback: Optional callback function(progress_percent, message)
        output_path: Optional specific path for the output file. If provided and exists, conversion is skipped.
    
    Returns:
        Path to converted video (or original if no conversion needed)
    """
    # Check if output_path is provided and exists
    if output_path and os.path.exists(output_path):
        print(f"Found cached 720p video at: {output_path}")
        if progress_callback:
            progress_callback(100, "Using cached 720p video")
        return output_path

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
    if min(width, height) <= 720:
        print(f"Video is already 720p (min dimension {min(width, height)}), no conversion needed")
        cap.release()
        if progress_callback:
            progress_callback(100, "Video ready (no conversion needed)")
        return input_path
    
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = width / height
    
    if width < height: # Portrait
        new_width = 720
        new_height = int(new_width / aspect_ratio)
    elif aspect_ratio > 16/9:  # Wider than 16:9
        new_width = 1280
        new_height = int(1280 / aspect_ratio)
    else:  # Taller than or equal to 16:9 (Landscape)
        new_height = 720
        new_width = int(720 * aspect_ratio)
    
    # Ensure dimensions are even (required for some codecs)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    print(f"Converting to: {new_width}x{new_height}")
    
    # Create output path
    if output_path:
        # Use provided output path
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    elif output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_720p_")
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_720p{ext}")
    else:
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_720p{ext}")
    
    print(f"Converting video to 720p using FFmpeg...")
    
    # Determine scaling parameter
    if width < height: # Portrait
        scale_param = "720:-2"
    else: # Landscape
        scale_param = "-2:720"
        
    # Construct FFmpeg command
    # -y: Overwrite output files
    # -i: Input file
    # -vf: Video filter (scaling)
    # -c:v: Video codec (libx264)
    # -preset: Encoding speed (fast/faster/veryfast)
    # -crf: Constant Rate Factor (quality, 23 is default)
    # -c:a: Audio codec (copy to avoid re-encoding audio if possible, or aac)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale={scale_param}",
        "-c:v", "libx264",
        "-preset", "faster",
        "-crf", "23",
        "-pix_fmt", "yuv420p", # Ensure browser compatibility
        "-c:a", "aac", 
        output_path
    ]
    
    try:
        if progress_callback:
            progress_callback(10, "Starting FFmpeg conversion...")
            
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Parse stderr for progress
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                # Parse frame= N
                match = re.search(r"frame=\s*(\d+)", line)
                if match:
                    current_frame = int(match.group(1))
                    if total_frames > 0:
                        progress = int((current_frame / total_frames) * 100)
                        # Update every 5% or so to avoid spamming
                        if progress_callback and (progress % 5 == 0 or progress == 100):
                             progress_callback(progress, f"Converting: {progress}%")

        # Wait for process to finish
        process.wait()
        
        if process.returncode != 0:
            # Read any remaining stderr
            stderr_output = process.stderr.read()
            print(f"FFmpeg error: {stderr_output}")
            raise RuntimeError(f"FFmpeg failed with exit code {process.returncode}")
            
        if progress_callback:
            progress_callback(100, "Conversion complete")
            
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        # Fallback to OpenCV if FFmpeg fails? 
        # For now, let's raise the error as the user specifically requested acceleration
        raise e

    return output_path
    


def process_video_file(video_path):
    """
    Process video file in background thread
    Updates global processing_status
    """
    global current_video_path, original_video_path, original_video_name
    global video_ready, processing_status, video_info, video_version
    
    try:
        # Store original video information
        original_video_path = os.path.abspath(video_path)
        original_video_name = os.path.basename(video_path)
        
        def update_progress(progress, message):
            """Callback to update processing status"""
            processing_status["progress"] = progress
            processing_status["message"] = message
        
        # Determine cache path in media directory
        media_dir = os.path.join(REPO_ROOT, 'media')
        os.makedirs(media_dir, exist_ok=True)
        
        name, ext = os.path.splitext(original_video_name)
        cache_path = os.path.join(media_dir, f"{name}_720p.mp4")
        
        # Convert video to 720p
        converted_path = convert_to_720p(video_path, progress_callback=update_progress, output_path=cache_path)
        current_video_path = converted_path
        
        # Get video info
        cap = cv2.VideoCapture(current_video_path)
        video_info = {
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()
        
        # Mark as ready
        video_ready = True
        video_version = int(time.time() * 1000)
        processing_status = {
            "status": "complete",
            "progress": 100,
            "message": "Video processing complete!"
        }
        
        print(f"✓ Video ready for annotation: {original_video_name}")
        
    except Exception as e:
        print(f"✗ Error processing video: {e}")
        processing_status = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }
        video_ready = False

def start_server(video_path=None, port=5555, skip_conversion=False):
    """
    Start the annotation server
    
    Args:
        video_path: Optional path to pre-load a video file (None = use web upload)
        port: Port number for the server (default: 5555)
        skip_conversion: Skip 720p conversion if True (only applies if video_path provided)
    """
    global current_video_path, original_video_path, original_video_name, video_ready, video_info, video_version
    
    print("=" * 60)
    print(f"📹 Professional Video Annotation Editor")
    print("=" * 60)
    
    # If video path provided, pre-process it
    if video_path:
        # Validate video path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Store original video information
        original_video_path = os.path.abspath(video_path)
        original_video_name = os.path.basename(video_path)
        
        print(f"Pre-loading video: {original_video_name}")
        print(f"Full path: {original_video_path}")
        print()
        
        # Convert video to 720p if needed
        if skip_conversion:
            print("⚠️  Skipping video conversion (using original)")
            current_video_path = original_video_path
        else:
            print("📹 Checking video resolution...")
            converted_path = convert_to_720p(video_path)
            current_video_path = converted_path
        
        # Get video info
        cap = cv2.VideoCapture(current_video_path)
        video_info = {
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()
        
        video_ready = True
        video_version = int(time.time() * 1000)
        print("✓ Video ready for annotation")
        print(f"Duration: {video_info['duration']:.1f}s")
    else:
        print("Mode: Upload via web interface")
        print("Users can upload videos or provide paths through the web UI")
    
    print()
    print("=" * 60)
    print("🚀 Starting server...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📍 Or: http://127.0.0.1:{port}")
    print()
    if not video_path:
        print("📤 Visit the URL to upload a video file")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        raise

def main():
    """Main entry point for standalone usage"""
    parser = argparse.ArgumentParser(
        description='Professional Video Annotation Editor - Standalone Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with web upload interface
  python video_editor_annotation_server.py
  
  # Pre-load a video file
  python video_editor_annotation_server.py video.mp4
  
  # Use custom port
  python video_editor_annotation_server.py --port 8080
  
  # Pre-load video and skip 720p conversion
  python video_editor_annotation_server.py video.mp4 --skip-conversion
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        nargs='?',  # Make it optional
        default=None,
        help='Optional: Path to pre-load a video file (or use web upload)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5555,
        help='Port number for the server (default: 5555)'
    )
    
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip 720p conversion and use original video (only if video_path provided)'
    )
    
    args = parser.parse_args()
    
    try:
        start_server(args.video_path, port=args.port, skip_conversion=args.skip_conversion)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
