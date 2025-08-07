#!/usr/bin/env python
"""
Video Annotation Server with Real Time Capture
Uses Flask to serve video with proper HTML5 player that exposes currentTime
"""

from flask import Flask, render_template_string, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import json
from datetime import datetime
import tempfile
import shutil

app = Flask(__name__)
CORS(app)

# Store annotations in memory (could use database)
annotations = []
current_video_path = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Annotator with Time Capture</title>
    <style>
        body {
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .video-section {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        video {
            width: 100%;
            max-height: 500px;
            border-radius: 8px;
        }
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .control-panel {
            background: #3d3d3d;
            padding: 20px;
            border-radius: 8px;
        }
        .time-display {
            font-size: 36px;
            font-weight: bold;
            color: #4ade80;
            font-family: 'Monaco', monospace;
            text-align: center;
            padding: 20px;
            background: #1a1a1a;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.2s;
        }
        button:hover {
            transform: scale(1.05);
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-success {
            background: #10b981;
            color: white;
        }
        .btn-warning {
            background: #f59e0b;
            color: white;
        }
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            background: #2d2d2d;
            border: 1px solid #4d4d4d;
            color: white;
            border-radius: 6px;
            font-size: 14px;
        }
        .annotations-table {
            width: 100%;
            background: #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #4d4d4d;
        }
        th {
            background: #3d3d3d;
            font-weight: 600;
        }
        .segment-times {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        .time-input-group {
            display: flex;
            flex-direction: column;
        }
        .capture-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }
        .keyboard-hint {
            background: #4d4d4d;
            padding: 10px;
            border-radius: 6px;
            margin-top: 20px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Video Annotation Tool with Real Time Capture</h1>
        
        <div class="video-section">
            <video id="videoPlayer" controls>
                <source src="{{ video_url }}" type="video/mp4">
            </video>
            
            <div class="time-display" id="currentTime">0:00.0</div>
        </div>
        
        <div class="controls">
            <div class="control-panel">
                <h3>⏱️ Time Capture</h3>
                
                <div class="capture-buttons">
                    <button class="btn-primary" onclick="captureTime()">📍 Capture</button>
                    <button class="btn-success" onclick="setAsStart()">⬅️ Set Start</button>
                    <button class="btn-warning" onclick="setAsEnd()">➡️ Set End</button>
                </div>
                
                <div class="segment-times">
                    <div class="time-input-group">
                        <label>Start Time (sec)</label>
                        <input type="number" id="startTime" step="0.1" value="0">
                    </div>
                    <div class="time-input-group">
                        <label>End Time (sec)</label>
                        <input type="number" id="endTime" step="0.1" value="5">
                    </div>
                </div>
                
                <button class="btn-primary" onclick="quickSegment()" style="width: 100%;">
                    ⚡ Quick 5-second Segment
                </button>
                
                <div class="keyboard-hint">
                    <strong>Keyboard Shortcuts:</strong><br>
                    Space: Play/Pause | C: Capture | S: Set Start | E: Set End | Q: Quick Segment
                </div>
            </div>
            
            <div class="control-panel">
                <h3>📝 Annotation Details</h3>
                
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
                    <option value="Other">Other</option>
                </select>
                
                <label>Side</label>
                <select id="side">
                    <option value="bilateral">Bilateral</option>
                    <option value="left">Left</option>
                    <option value="right">Right</option>
                    <option value="n/a">N/A</option>
                </select>
                
                <label>Severity (0-4)</label>
                <input type="number" id="severity" min="0" max="4" value="0">
                
                <label>Notes</label>
                <textarea id="notes" rows="3" placeholder="Optional notes..."></textarea>
                
                <button class="btn-success" onclick="addAnnotation()" style="width: 100%; margin-top: 15px;">
                    ➕ Add Annotation
                </button>
            </div>
        </div>
        
        <div class="annotations-table">
            <h3>📋 Annotations</h3>
            <table id="annotationsTable">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Start</th>
                        <th>End</th>
                        <th>Duration</th>
                        <th>Task</th>
                        <th>Side</th>
                        <th>Severity</th>
                        <th>Notes</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody id="annotationsBody">
                </tbody>
            </table>
            
            <div style="margin-top: 20px;">
                <button class="btn-primary" onclick="saveAnnotations()">💾 Save All</button>
                <button class="btn-danger" onclick="clearAnnotations()">🗑️ Clear All</button>
                <button class="btn-success" onclick="exportJSON()">📥 Export JSON</button>
            </div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('videoPlayer');
        const timeDisplay = document.getElementById('currentTime');
        let capturedTime = 0;
        
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
            
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    video.paused ? video.play() : video.pause();
                    break;
                case 'c':
                case 'C':
                    captureTime();
                    break;
                case 's':
                case 'S':
                    setAsStart();
                    break;
                case 'e':
                case 'E':
                    setAsEnd();
                    break;
                case 'q':
                case 'Q':
                    quickSegment();
                    break;
            }
        });
        
        function captureTime() {
            capturedTime = video.currentTime;
            timeDisplay.style.background = '#065f46';
            setTimeout(() => { timeDisplay.style.background = '#1a1a1a'; }, 300);
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
            
            fetch('/add_annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(annotation)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadAnnotations();
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
                    data.annotations.forEach((ann, idx) => {
                        const row = tbody.insertRow();
                        row.innerHTML = `
                            <td>${idx}</td>
                            <td>${ann.start.toFixed(1)}</td>
                            <td>${ann.end.toFixed(1)}</td>
                            <td>${(ann.end - ann.start).toFixed(1)}</td>
                            <td>${ann.task}</td>
                            <td>${ann.side}</td>
                            <td>${ann.severity}</td>
                            <td>${ann.notes}</td>
                            <td><button class="btn-danger" onclick="deleteAnnotation(${idx})">Delete</button></td>
                        `;
                    });
                });
        }
        
        function deleteAnnotation(idx) {
            fetch(`/delete_annotation/${idx}`, {method: 'DELETE'})
                .then(() => loadAnnotations());
        }
        
        function clearAnnotations() {
            if (confirm('Clear all annotations?')) {
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
    filename = f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({
            'video': current_video_path,
            'annotations': annotations,
            'created': datetime.now().isoformat()
        }, f, indent=2)
    return jsonify({'success': True, 'message': f'Saved to {filename}'})

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
    global current_video_path
    current_video_path = video_path
    print(f"Starting annotation server for video: {video_path}")
    print("Open http://localhost:5555 in your browser")
    app.run(host='0.0.0.0', port=5555, debug=False)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        start_server(sys.argv[1])
    else:
        print("Usage: python video_annotation_server.py <video_path>")