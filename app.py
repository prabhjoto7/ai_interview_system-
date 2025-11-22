"""
Complete AI Interview System - Flask + HTML/JS Camera
Deployable on: Heroku, Railway, Render, PythonAnywhere
"""

from flask import Flask, render_template, request, jsonify, session, send_file
import base64
import os
import time
import json
import tempfile
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
import threading

from queue import Queue
# AI/ML Imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    DEEPFACE_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except:
    MP_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except:
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except:
    SR_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False

app = Flask(__name__)
# app.secret_key = 'your-secret-key-change-in-production'
import os
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
# ==================== CONFIGURATION ====================

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

# Interview Questions
QUESTIONS = [
    {
        "id": 1,
        "question": "Tell me about yourself.",
        "type": "personal",
        "duration": 60,
        "ideal_answer": "I'm a computer science postgraduate with strong interest in AI and software development. I've worked on projects involving Python, machine learning, and data analysis.",
        "tip": "Focus on your background, skills, and personality"
    },
    {
        "id": 2,
        "question": "What are your strengths and weaknesses?",
        "type": "behavioral",
        "duration": 60,
        "ideal_answer": "My strength is being detail-oriented and persistent. My weakness is that I sometimes spend too much time perfecting details, but I'm working on prioritizing better.",
        "tip": "Be honest and show self-awareness"
    },
    {
        "id": 3,
        "question": "Where do you see yourself in 5 years?",
        "type": "career",
        "duration": 60,
        "ideal_answer": "I see myself growing into a senior role where I can contribute to meaningful AI projects and mentor junior team members.",
        "tip": "Show ambition aligned with career growth"
    }
]

# Violation thresholds
VIOLATION_THRESHOLD = 3  # Auto-terminate after 3 violations
MIN_INTEGRITY_SCORE = 40  # Terminate if below this

# ==================== MODEL LOADING ====================

class ModelManager:
    """Singleton model manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models = {}
        self._load_models()
        self._initialized = True
    
    def _load_models(self):
        """Load all AI models once"""
        print("🔄 Loading AI models...")
        
        # DeepFace (auto-loads on first use)
        if DEEPFACE_AVAILABLE:
            try:
                _ = DeepFace.build_model("Facenet")
                self.models['deepface'] = True
                print("✅ DeepFace loaded")
            except:
                self.models['deepface'] = False
                print("⚠️ DeepFace failed")
        
        # MediaPipe
        if MP_AVAILABLE:
            try:
                self.models['face_mesh'] = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
                self.models['hands'] = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                print("✅ MediaPipe loaded")
            except:
                self.models['face_mesh'] = None
                self.models['hands'] = None
                print("⚠️ MediaPipe failed")
        
        # YOLO
        if YOLO_AVAILABLE:
            try:
                self.models['yolo'] = YOLO("yolov8n.pt")
                self.models['yolo_cls'] = YOLO("yolov8n-cls.pt")
                print("✅ YOLO loaded")
            except:
                self.models['yolo'] = None
                self.models['yolo_cls'] = None
                print("⚠️ YOLO failed")
        
        # SentenceTransformer
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.models['sentence'] = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ SentenceTransformer loaded")
            except:
                self.models['sentence'] = None
                print("⚠️ SentenceTransformer failed")
model_manager = None

def get_model_manager():
    """Lazy load models on first request"""
    global model_manager
    if model_manager is None:
        print("🔄 Initializing model manager...")
        model_manager = ModelManager()
        print("✅ Model manager ready")
    return model_manager

# ==================== ANALYSIS FUNCTIONS ====================

class FrameAnalyzer:
    """Analyzes frames for violations and quality"""
    
    def __init__(self):
        self.models = get_model_manager().models
        self.blink_history = []  # Track blink state
        self.last_blink_state = False
    
    def detect_blink(self, face_landmarks):
        """Detect eye blink using EAR (Eye Aspect Ratio)"""
        try:
            # Left eye indices
            left_eye = [33, 160, 158, 133, 153, 144]
            # Right eye indices
            right_eye = [362, 385, 387, 263, 373, 380]
            
            def eye_aspect_ratio(eye_points):
                landmarks = face_landmarks.landmark
                # Vertical distances
                v1 = np.linalg.norm(np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]) - 
                                   np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y]))
                v2 = np.linalg.norm(np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]) - 
                                   np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y]))
                # Horizontal distance
                h = np.linalg.norm(np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]) - 
                                  np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y]))
                
                ear = (v1 + v2) / (2.0 * h)
                return ear
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # EAR threshold for blink detection
            return avg_ear < 0.21
        except:
            return False
    
    def analyze_frame(self, frame_data):
        """
        Analyze a single frame
        Returns: dict with violations, metrics, and warnings
        """
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = {
            'timestamp': time.time(),
            'violations': [],
            'warnings': [],
            'metrics': {},
            'face_detected': False,
            'multiple_people': False,
            'cheating_items': [],
            'lighting_quality': 'good',
            'blink_detected': False,
            'eye_contact': False
        }
        
        # 1. Face Detection with Blink & Eye Contact
        num_faces = 0
        face_box = None
        
        if self.models.get('face_mesh'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.models['face_mesh'].process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                num_faces = len(face_results.multi_face_landmarks)
                result['face_detected'] = True
                
                if num_faces > 1:
                    result['violations'].append({
                        'type': 'multiple_people',
                        'severity': 'high',
                        'message': f'{num_faces} faces detected'
                    })
                    result['multiple_people'] = True
                
                # Get face bounding box and analyze single face
                if num_faces == 1:
                    landmarks = face_results.multi_face_landmarks[0]
                    h, w = frame.shape[:2]
                    x_coords = [lm.x * w for lm in landmarks.landmark]
                    y_coords = [lm.y * h for lm in landmarks.landmark]
                    face_box = (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords))
                    )
                    
                    # Blink detection
                    is_blinking = self.detect_blink(landmarks)
                    if is_blinking and not self.last_blink_state:
                        result['blink_detected'] = True
                    self.last_blink_state = is_blinking
                    
                    # Eye contact detection (simplified)
                    # Check if face is looking forward (nose tip vs face center)
                    nose_tip = landmarks.landmark[1]
                    face_center_y = np.mean(y_coords) / h
                    face_center_x = np.mean(x_coords) / w
                    
                    # If nose is roughly centered, assume eye contact
                    x_deviation = abs(nose_tip.x - face_center_x)
                    y_deviation = abs(nose_tip.y - face_center_y)
                    
                    if x_deviation < 0.15 and y_deviation < 0.15:
                        result['eye_contact'] = True
                    else:
                        result['warnings'].append({
                            'type': 'looking_away',
                            'message': 'Candidate appears to be looking away'
                        })
            else:
                result['warnings'].append({
                    'type': 'no_face',
                    'message': 'No face detected in frame'
                })
        
        # 2. Hand Detection (potential device usage)
        if self.models.get('hands'):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.models['hands'].process(rgb_frame)
                
                if hand_results.multi_hand_landmarks:
                    num_hands = len(hand_results.multi_hand_landmarks)
                    
                    # Check for suspicious hand positions (below frame, at ears)
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        
                        # Hand near ear (potential phone usage)
                        if face_box and wrist.y < 0.3:  # Upper part of frame
                            result['warnings'].append({
                                'type': 'suspicious_hand',
                                'message': 'Hand detected near head (potential device usage)'
                            })
                        
                        # Hand below visible area (potential typing)
                        if wrist.y > 0.8:
                            result['warnings'].append({
                                'type': 'suspicious_hand',
                                'message': 'Hand below frame (potential device usage)'
                            })
            except:
                pass
        
        # 3. Object Detection (cheating items)
        if self.models.get('yolo'):
            try:
                detections = self.models['yolo'].predict(frame, conf=0.4, verbose=False)
                if detections and len(detections) > 0:
                    names = self.models['yolo'].names
                    boxes = detections[0].boxes
                    
                    cheating_items = ['cell phone', 'book', 'laptop', 'tablet']
                    
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        obj_name = names[cls_id]
                        
                        if obj_name.lower() in cheating_items:
                            result['violations'].append({
                                'type': 'cheating_item',
                                'severity': 'critical',
                                'message': f'Detected: {obj_name}',
                                'object': obj_name
                            })
                            result['cheating_items'].append(obj_name)
            except:
                pass
        
        # 4. Lighting Analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 60:
            result['lighting_quality'] = 'too_dark'
            result['warnings'].append({
                'type': 'lighting',
                'message': 'Lighting is too dark'
            })
        elif brightness > 200:
            result['lighting_quality'] = 'too_bright'
            result['warnings'].append({
                'type': 'lighting',
                'message': 'Lighting is too bright'
            })
        
        result['metrics']['brightness'] = float(brightness)
        result['metrics']['num_faces'] = num_faces
        
        return result

frame_analyzer = None

def get_frame_analyzer():
    """Lazy load frame analyzer"""
    global frame_analyzer
    if frame_analyzer is None:
        print("🔄 Initializing frame analyzer...")
        frame_analyzer = FrameAnalyzer()
        print("✅ Frame analyzer ready")
    return frame_analyzer

# ==================== SESSION MANAGER ====================

class InterviewSession:
    """Manages interview session state"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.start_time = time.time()
        self.frames = []
        self.violations = []
        self.current_question = 0
        self.answers = []
        self.integrity_score = 100
        self.status = 'active' 
        self.tab_switches = 0 # active, paused, completed, terminated
        self.metadata = {
            'total_frames': 0,
            'frames_with_face': 0,
            'frames_no_face': 0,
            'total_violations': 0
        }
    
    def add_frame_result(self, result):
        """Process frame analysis result"""
        self.frames.append(result)
        self.metadata['total_frames'] += 1
        
        if result['face_detected']:
            self.metadata['frames_with_face'] += 1
        else:
            self.metadata['frames_no_face'] += 1
        
        # Add violations
        for violation in result['violations']:
            self.violations.append({
                **violation,
                'timestamp': result['timestamp'],
                'question': self.current_question
            })
            self.metadata['total_violations'] += 1
            
            # Reduce integrity score
            if violation['severity'] == 'critical':
                self.integrity_score = max(0, self.integrity_score - 10)
            elif violation['severity'] == 'high':
                self.integrity_score = max(0, self.integrity_score - 5)
        
        # Check termination conditions
        if len(self.violations) >= VIOLATION_THRESHOLD:
            self.status = 'terminated'
            self.termination_reason = f'{len(self.violations)} violations detected'
        
        if self.integrity_score < MIN_INTEGRITY_SCORE:
            self.status = 'terminated'
            self.termination_reason = f'Integrity score fell to {self.integrity_score}%'
    
    def next_question(self):
        """Move to next question"""
        self.current_question += 1
        if self.current_question >= len(QUESTIONS):
            self.status = 'completed'
    
    def get_summary(self):
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'duration': time.time() - self.start_time,
            'status': self.status,
            'integrity_score': self.integrity_score,
            'total_violations': self.metadata['total_violations'],
            'questions_answered': len(self.answers),
            'current_question': self.current_question,
            'metadata': self.metadata,
            'tab_switches': self.tab_switches
        }

# Global sessions store
sessions = {}

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main interview page"""
    # Initialize new session
    session_id = f"session_{int(time.time())}_{os.urandom(4).hex()}"
    session['session_id'] = session_id
    sessions[session_id] = InterviewSession(session_id)
    
    return render_template('interview.html', 
                         questions=QUESTIONS,
                         session_id=session_id)

@app.route('/capture', methods=['POST'])
def capture_frame():
    """Receive and analyze frame from frontend"""
    data = request.get_json()
    session_id = session.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = sessions[session_id]
    
    # Analyze frame (lazy load analyzer)
    analyzer = get_frame_analyzer()
    result = analyzer.analyze_frame(data['image'])
    
    # Update session
    interview_session.add_frame_result(result)
    
    # Prepare response
    response = {
        'status': interview_session.status,
        'integrity_score': interview_session.integrity_score,
        'violations': result['violations'],
        'warnings': result['warnings'],
        'should_terminate': interview_session.status == 'terminated'
    }
    
    if interview_session.status == 'terminated':
        response['reason'] = interview_session.termination_reason
    
    return jsonify(response)
@app.route('/setup_complete', methods=['POST'])
def setup_complete():
    """Handle setup completion signal from frontend"""
    return jsonify({'status': 'success'})
# ========================================
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Submit answer (audio + metadata)"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = sessions[session_id]
    
    data = request.get_json()
    
    # Save audio if provided
    audio_path = None
    if 'audio' in data:
        audio_data = base64.b64decode(data['audio'].split(',')[1])
        audio_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_q{data['question_id']}.wav")
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
    
    # Store answer
    answer = {
        'question_id': data['question_id'],
        'question': data['question_text'],
        'audio_path': audio_path,
        'transcript': data.get('transcript', ''),
        'duration': data.get('duration', 0),
        'timestamp': time.time()
    }
    
    interview_session.answers.append(answer)
    interview_session.next_question()
    
    return jsonify({
        'status': 'success',
        'next_question': interview_session.current_question,
        'session_status': interview_session.status
    })

@app.route('/get_status')
def get_status():
    """Get current session status"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'No active session'}), 400
    
    interview_session = sessions[session_id]
    return jsonify(interview_session.get_summary())

@app.route('/analyze', methods=['POST'])
def analyze_interview():
    """Phase 2: Analyze complete interview"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = sessions[session_id]
    
    # Run comprehensive analysis
    analysis_results = run_post_interview_analysis(interview_session)
    
    return jsonify(analysis_results)

@app.route('/results')
def results_page():
    """Display final results"""
    session_id = session.get('session_id')
    
    if not session_id or session_id not in sessions:
        return "No active session", 400
    
    interview_session = sessions[session_id]
    analysis = run_post_interview_analysis(interview_session)
    
    return render_template('results.html', 
                         session=interview_session.get_summary(),
                         analysis=analysis)
@app.route('/log_tab_switch', methods=['POST'])
def log_tab_switch():
    session_id = session.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    # Update the count
    sessions[session_id].tab_switches += 1
    
    # Penalty: -2 points for switching tabs
    sessions[session_id].integrity_score = max(0, sessions[session_id].integrity_score - 2)
    
    return jsonify({
        'status': 'success',
        'tab_switches': sessions[session_id].tab_switches,
        'integrity_score': sessions[session_id].integrity_score
    })
# ==================== PHASE 2 ANALYSIS ====================

def run_post_interview_analysis(interview_session):
    """
    Comprehensive post-interview analysis
    Returns hiring decision with detailed metrics
    """
    results = {
        'overall_score': 0,
        'confidence': 0,
        'nervousness': 0,
        'fluency': 0,
        'answer_accuracy': 0,
        'integrity_score': interview_session.integrity_score,
        'questions': [],
        'decision': 'No Hire',
        'reasons': [],
        'eye_contact_percentage': 0,
        'blink_rate': 0
    }
    
    # Analyze each answer
    for answer in interview_session.answers:
        # Find the question object that matches this answer
        try:
            question = next(q for q in QUESTIONS if q['id'] == answer['question_id'])
        except StopIteration:
            continue # Skip if question ID not found

        question_analysis = {
            'question': question['question'],
            'transcript': answer.get('transcript', ''),
            'metrics': {}
        }
        
        # 1. Fluency Analysis
        if answer.get('transcript'):
            words = answer['transcript'].split()
            duration = answer.get('duration', 0)
            wpm = (len(words) / duration) * 60 if duration > 0 else 0
            question_analysis['metrics']['wpm'] = wpm
            question_analysis['metrics']['word_count'] = len(words)
        
        # 2. Answer Accuracy (semantic similarity)
        # Check if model exists AND transcript exists
        if get_model_manager().models.get('sentence') and answer.get('transcript'):
            try:
                embeddings = get_model_manager().models['sentence'].encode([
                    question['ideal_answer'],
                    answer['transcript']
                ])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                question_analysis['metrics']['accuracy'] = float(similarity * 100)
            except:
                question_analysis['metrics']['accuracy'] = 50
        else:
            # Default score if model is missing (e.g. Free Tier)
            question_analysis['metrics']['accuracy'] = 0
        
        results['questions'].append(question_analysis)
    
    # Calculate overall metrics
    if results['questions']:
        # Filter out 0s to avoid dragging down average if models failed
        accuracies = [q['metrics'].get('accuracy', 0) for q in results['questions']]
        results['answer_accuracy'] = np.mean(accuracies) if accuracies else 0
        
        wpms = [q['metrics'].get('wpm', 0) for q in results['questions']]
        results['fluency'] = np.mean(wpms) if wpms else 0
    
    # === FIX: Calculate Eye Contact & Blinks ===
    # Get all frames where a face was actually seen
    face_frames = [f for f in interview_session.frames if f.get('face_detected')]
    total_face_frames = len(face_frames)
    
    if total_face_frames > 0:
        # Count frames with eye contact
        eye_contact_count = sum(1 for f in face_frames if f.get('eye_contact'))
        results['eye_contact_percentage'] = round((eye_contact_count / total_face_frames) * 100, 1)
        
        # Count blinks (simple approximation)
        blink_count = sum(1 for f in face_frames if f.get('blink_detected'))
        # Approximate blink rate per minute based on session duration
        duration_min = (time.time() - interview_session.start_time) / 60
        if duration_min > 0:
            results['blink_rate'] = round(blink_count / duration_min, 1)
    
    # Simplified emotion scores
    results['confidence'] = max(0, 80 - (len(interview_session.violations) * 5))
    results['nervousness'] = min(100, 20 + (len(interview_session.violations) * 5))
    
    # Calculate overall score
    results['overall_score'] = (
        results['confidence'] * 0.25 +
        results['answer_accuracy'] * 0.30 +
        results['fluency'] * 0.20 +
        results['integrity_score'] * 0.25
    )
    
    # Make hiring decision
    if results['overall_score'] >= 75 and interview_session.status == 'completed':
        results['decision'] = 'Strong Hire'
        results['reasons'].append('Excellent overall performance')
    elif results['overall_score'] >= 60:
        results['decision'] = 'Hire'
        results['reasons'].append('Good performance with minor concerns')
    elif results['overall_score'] >= 45:
        results['decision'] = 'Maybe'
        results['reasons'].append('Moderate performance')
    else:
        results['decision'] = 'No Hire'
        results['reasons'].append('Below expected standards')
    
    if interview_session.status == 'terminated':
        results['decision'] = 'Disqualified'
        results['reasons'].insert(0, f'Interview terminated: {getattr(interview_session, "termination_reason", "Unknown reason")}')
    
    if len(interview_session.violations) > 0:
        results['reasons'].append(f'{len(interview_session.violations)} integrity violations detected')
    
    return results
@app.route('/health')
def health():
    """Health check endpoint for Render"""
    # Don't load models during health check
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'ready': True
    })
# ==================== RUN ====================

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)