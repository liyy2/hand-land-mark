"""
Custom Video Player Component for Gradio with Time Capture
Provides a video player with exposed currentTime API for annotation
"""

import gradio as gr
import base64
import os

class VideoPlayerWithTime:
    """Custom video player that exposes current time for annotation"""
    
    @staticmethod
    def create_player_html(video_path=None, video_base64=None):
        """Create HTML for video player with time capture capabilities"""
        
        if not video_path:
            return """
            <div style="padding: 20px; background: #f0f0f0; border-radius: 8px; text-align: center;">
                <p>No video loaded. Please upload a video in the Video Processing tab.</p>
            </div>
            """
        
        # For Gradio, we need to encode the video as base64 or use a proxy URL
        if video_base64:
            video_url = f"data:video/mp4;base64,{video_base64}"
        else:
            # Try to read and encode the video
            try:
                with open(video_path, 'rb') as video_file:
                    video_data = video_file.read()
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                    video_url = f"data:video/mp4;base64,{video_base64}"
            except:
                # Fallback to file path (might not work in all browsers)
                video_url = f"file://{os.path.abspath(video_path)}"
        
        html = f"""
        <div id="custom-video-container" style="width: 100%; background: #000; border-radius: 8px; padding: 10px;">
            <video id="annotation-video" controls style="width: 100%; max-height: 500px;">
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            
            <div style="background: #2d3748; color: white; padding: 15px; margin-top: 10px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 18px; font-weight: bold;">Current Time: </span>
                        <span id="current-time-display" style="font-size: 24px; color: #4ade80; font-family: monospace;">0.0s</span>
                    </div>
                    <div>
                        <button id="capture-time-btn" onclick="captureTime()" 
                                style="padding: 10px 20px; background: #667eea; color: white; border: none; 
                                       border-radius: 6px; cursor: pointer; font-size: 16px;">
                            📍 Capture Time
                        </button>
                    </div>
                </div>
                
                <div style="margin-top: 15px;">
                    <div style="background: #1a202c; padding: 10px; border-radius: 6px;">
                        <label style="display: block; margin-bottom: 5px; color: #a0aec0;">Captured Time (seconds):</label>
                        <input id="captured-time-input" type="number" step="0.1" value="0" 
                               style="width: 100%; padding: 8px; background: #2d3748; color: white; 
                                      border: 1px solid #4a5568; border-radius: 4px; font-size: 16px;">
                    </div>
                </div>
                
                <div style="margin-top: 15px; display: flex; gap: 10px;">
                    <button onclick="setAsStart()" 
                            style="flex: 1; padding: 10px; background: #10b981; color: white; 
                                   border: none; border-radius: 6px; cursor: pointer;">
                        ⬅️ Set as Start
                    </button>
                    <button onclick="setAsEnd()" 
                            style="flex: 1; padding: 10px; background: #f59e0b; color: white; 
                                   border: none; border-radius: 6px; cursor: pointer;">
                        ➡️ Set as End
                    </button>
                    <button onclick="quickSegment()" 
                            style="flex: 1; padding: 10px; background: #8b5cf6; color: white; 
                                   border: none; border-radius: 6px; cursor: pointer;">
                        ⚡ Quick 5s
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            let video = null;
            let lastCapturedTime = 0;
            
            // Initialize when DOM is ready
            document.addEventListener('DOMContentLoaded', function() {{
                video = document.getElementById('annotation-video');
                if (video) {{
                    // Update time display continuously
                    video.addEventListener('timeupdate', updateTimeDisplay);
                    
                    // Keyboard shortcuts
                    document.addEventListener('keydown', function(e) {{
                        if (e.target.tagName !== 'INPUT') {{
                            if (e.key === ' ') {{
                                e.preventDefault();
                                togglePlayPause();
                            }} else if (e.key === 'c' || e.key === 'C') {{
                                captureTime();
                            }} else if (e.key === 's' || e.key === 'S') {{
                                setAsStart();
                            }} else if (e.key === 'e' || e.key === 'E') {{
                                setAsEnd();
                            }}
                        }}
                    }});
                }}
            }});
            
            function updateTimeDisplay() {{
                if (video) {{
                    const display = document.getElementById('current-time-display');
                    if (display) {{
                        const time = video.currentTime;
                        const minutes = Math.floor(time / 60);
                        const seconds = (time % 60).toFixed(1);
                        display.textContent = minutes > 0 ? minutes + ':' + seconds.padStart(4, '0') : seconds + 's';
                    }}
                }}
            }}
            
            function captureTime() {{
                if (video) {{
                    lastCapturedTime = video.currentTime;
                    const input = document.getElementById('captured-time-input');
                    if (input) {{
                        input.value = lastCapturedTime.toFixed(1);
                        input.style.background = '#065f46';
                        setTimeout(() => {{ input.style.background = '#2d3748'; }}, 300);
                    }}
                    
                    // Also update Gradio components if they exist
                    updateGradioTime(lastCapturedTime);
                }}
            }}
            
            function togglePlayPause() {{
                if (video) {{
                    if (video.paused) {{
                        video.play();
                    }} else {{
                        video.pause();
                    }}
                }}
            }}
            
            function setAsStart() {{
                captureTime();
                // Update start time in Gradio
                const startInput = parent.document.querySelector('input[aria-label*="Start Time"]');
                if (startInput) {{
                    startInput.value = lastCapturedTime.toFixed(1);
                    startInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }}
            
            function setAsEnd() {{
                captureTime();
                // Update end time in Gradio
                const endInput = parent.document.querySelector('input[aria-label*="End Time"]');
                if (endInput) {{
                    endInput.value = lastCapturedTime.toFixed(1);
                    endInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }}
            
            function quickSegment() {{
                captureTime();
                const start = Math.max(0, lastCapturedTime - 2.5);
                const end = lastCapturedTime + 2.5;
                
                const startInput = parent.document.querySelector('input[aria-label*="Start Time"]');
                const endInput = parent.document.querySelector('input[aria-label*="End Time"]');
                
                if (startInput && endInput) {{
                    startInput.value = start.toFixed(1);
                    startInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    endInput.value = end.toFixed(1);
                    endInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }}
            
            function updateGradioTime(time) {{
                // Try to update the Gradio current time input
                const gradioInput = parent.document.querySelector('input[aria-label*="Current Video Time"]');
                if (gradioInput) {{
                    gradioInput.value = time.toFixed(1);
                    gradioInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }}
        </script>
        
        <style>
            #custom-video-container button:hover {{
                opacity: 0.9;
                transform: scale(1.02);
                transition: all 0.2s;
            }}
            
            #custom-video-container button:active {{
                transform: scale(0.98);
            }}
        </style>
        """
        
        return html
    
    @staticmethod
    def create_simple_player(video_path):
        """Create a simpler player using Gradio's HTML component"""
        
        if not video_path:
            return "<p>No video loaded</p>", 0, 0
        
        return f"""
        <div style="width: 100%;">
            <video id="anno-video" controls style="width: 100%;" src="file={video_path}">
            </video>
            <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                <button onclick="
                    var v = document.getElementById('anno-video');
                    var t = v ? v.currentTime : 0;
                    document.getElementById('time-out').value = t;
                    document.getElementById('time-out').dispatchEvent(new Event('change'));
                " style="padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 4px;">
                    Capture Time
                </button>
                <input type="hidden" id="time-out" />
                <span id="time-display" style="margin-left: 10px; font-weight: bold;">Time: 0s</span>
            </div>
            <script>
                setInterval(function() {{
                    var v = document.getElementById('anno-video');
                    if (v) {{
                        document.getElementById('time-display').textContent = 'Time: ' + v.currentTime.toFixed(1) + 's';
                    }}
                }}, 100);
            </script>
        </div>
        """, 0, 0