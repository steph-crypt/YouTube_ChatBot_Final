// App.jsx - Full Voice + TTS Support
import { useState, useRef } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    const chunks = [];

    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/wav" });
      await sendAudio(blob);
    };

    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current?.stream.getTracks().forEach(t => t.stop());
    setRecording(false);
  };

  const sendAudio = async (audioBlob) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("audio", audioBlob, "question.wav");

    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setQuestion(data.question || "");
    setAnswer(data.answer);
    setAudioUrl(data.audio_url);
    setLoading(false);

    // Auto-play response
    if (audioRef.current && data.audio_url) {
      audioRef.current.play();
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("question", question);

    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setAnswer(data.answer);
    setAudioUrl(data.audio_url);
    setLoading(false);

    if (audioRef.current && data.audio_url) {
      audioRef.current.play();
    }
  };

  return (
    <div style={{ maxWidth: 700, margin: "50px auto", fontFamily: "sans-serif" }}>
      <h1>Chat Degrass Tyson</h1>
      <p>Ask me anything about science — with your voice!</p>

      <div style={{ margin: "20px 0" }}>
        <button
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onTouchStart={startRecording}
          onTouchEnd={stopRecording}
          disabled={loading}
          style={{
            padding: "15px 30px",
            fontSize: "18px",
            background: recording ? "#ff4444" : "#0066ff",
            color: "white",
            border: "none",
            borderRadius: "50px",
            cursor: "pointer",
          }}
        >
          {recording ? "Release to Send" : "Hold to Speak"}
        </button>
      </div>

      <textarea
        placeholder="Or type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={3}
        style={{ width: "100%", padding: 12, fontSize: "16px" }}
      />

      <button onClick={askQuestion} disabled={loading || !question.trim()}>
        {loading ? "Thinking..." : "Ask"}
      </button>

      {answer && (
        <div style={{ marginTop: 30, padding: 20, background: "#f0f8ff", borderRadius: 10 }}>
          <h3>Answer:</h3>
          <p style={{ fontSize: "18px", lineHeight: "1.6" }}>{answer}</p>
          
          {audioUrl && (
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              autoPlay
              style={{ width: "100%", marginTop: 15 }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;