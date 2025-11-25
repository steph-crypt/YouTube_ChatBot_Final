import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  async function askQuestion() {
    setLoading(true);
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    setAnswer(data.answer);
    setLoading(false);
  }

  return (
    <div style={{ maxWidth: 600, margin: "50px auto", fontFamily: "sans-serif" }}>
      <h1>Chat Degrass Tyson</h1>

      <textarea
        placeholder="Ask a question..."
        value={question}
        onChange={e => setQuestion(e.target.value)}
        rows={4}
        style={{ width: "100%", padding: 10 }}
      />

      <button onClick={askQuestion} disabled={loading}>
        {loading ? "Thinking..." : "Ask"}
      </button>

      {answer && (
        <div style={{ marginTop: 20, padding: 10, background: "#fafafa" }}>
          <h3>Answer:</h3>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
