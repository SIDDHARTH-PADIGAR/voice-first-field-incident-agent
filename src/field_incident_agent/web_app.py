"""Local-only browser demo for synthetic field incident intake."""

from dataclasses import asdict
from pathlib import Path
import secrets
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from field_incident_agent.conversation import ConversationManager
from field_incident_agent.cli import (
    DEMO_PRINCIPAL_ID,
    DEMO_SITE_ID,
    _ensure_demo_assets,
)
from field_incident_agent.incident_service import IncidentValidationError
from field_incident_agent.llm_client import ModelClientError
from field_incident_agent.registry import AccessContext
from field_incident_agent.runtime import IncidentRuntime, build_runtime


DEFAULT_DATABASE = Path("data/demo/incident_demo.sqlite3")
MAX_PENDING_PROPOSALS = 100


class IntakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000)


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: bool


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: str | None = Field(default=None, max_length=100)
    text: str = Field(min_length=1, max_length=5000)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Field Incident Intake</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { max-width: 850px; margin: 32px auto; padding: 0 18px;
       background: #111827; color: #e5e7eb; line-height: 1.5; }
h1 { margin-bottom: 4px; }
.note { color: #cbd5e1; font-size: .95rem; }
.card { background: #1f2937; border: 1px solid #374151;
        border-radius: 12px; padding: 18px; margin: 18px 0; }
textarea { box-sizing: border-box; width: 100%; min-height: 95px;
           padding: 12px; border: 1px solid #4b5563; border-radius: 8px;
           background: #111827; color: #f9fafb; font: inherit; }
button, select { font: inherit; border-radius: 7px; padding: 10px 14px; }
button { margin: 5px 5px 5px 0; border: 0; cursor: pointer; }
button:disabled { opacity: .55; cursor: wait; }
.primary { background: #60a5fa; color: #111827; font-weight: 650; }
.confirm { background: #86efac; color: #052e16; font-weight: 700; }
#chatLog { display: flex; flex-direction: column; gap: 12px; }
.bubble { white-space: pre-wrap; overflow-wrap: anywhere; padding: 13px 15px;
          border-radius: 12px; max-width: 92%; }
.agent { align-self: flex-start; background: #374151; }
.caller { align-self: flex-end; background: #1d4ed8; color: white; }
.speaker { font-size: .78rem; font-weight: 700; opacity: .8; margin-bottom: 3px; }
.small { font-size: .85rem; color: #9ca3af; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: .9rem; }
label { display: inline-block; margin: 8px 0; }
</style>
</head>
<body>
<h1>Field Incident Intake</h1>
<p class="note">Local development demo · Synthetic assets only</p>

<div class="card">
  <p><strong>Demo assets:</strong> DEMO-PUMP-001, DEMO-PUMP-002,
  DEMO-LIFT-001, DEMO-PANEL-001</p>
  <p class="small">Use synthetic reports only. Browser speech recognition may
  send audio to the browser vendor's recognition service. Do not speak real
  names, personal data, or sensitive incidents into this prototype.</p>
  <label for="report"><strong>Your message</strong></label>
  <textarea id="report" maxlength="5000"
    placeholder="Describe the incident, answer a question, or correct a detail..."></textarea>
  <div>
    <button id="startVoice" type="button">Start voice input</button>
    <button id="stopVoice" type="button" disabled>Stop listening</button>
    <button id="sendMessage" class="primary" type="button">Send message</button>
  </div>
  <p id="voiceStatus" class="small" aria-live="polite">
    You can speak or type. A completed voice turn is sent automatically.
  </p>
</div>

<div class="card">
  <h2>Conversation</h2>
  <div id="chatLog" role="log" aria-live="polite">
    <div class="bubble agent">
      <div class="speaker">Agent</div>
      Tell me what happened. I'll identify the equipment, summarize my
      understanding, and let you correct details before any normal ticket
      is created.
    </div>
  </div>

  <p>
    <label for="voiceSelect"><strong>Spoken voice</strong></label>
    <select id="voiceSelect" aria-label="Spoken voice"></select>
  </p>
  <p>
    <label><input id="autoSpeak" type="checkbox" checked>
      Speak replies automatically</label>
  </p>
  <button id="repeatResponse" type="button">Repeat last response</button>
  <button id="confirm" class="confirm" type="button" hidden>
    Confirm and create ticket
  </button>

  <details id="developerDetails">
    <summary>Technical details (developer view)</summary>
    <pre id="details">No extraction yet.</pre>
  </details>
</div>

<p class="small">Development prototype only. Tickets are local SQLite records,
not external maintenance tickets. Conversation and pending confirmation state
are held in server memory and do not survive a server restart. The demo uses
a synthetic identity and is not suitable for public deployment.</p>

<script>
(() => {
  const reportInput = document.getElementById("report");
  const sendButton = document.getElementById("sendMessage");
  const confirmButton = document.getElementById("confirm");
  const chatLog = document.getElementById("chatLog");
  const voiceStatus = document.getElementById("voiceStatus");
  const voiceSelect = document.getElementById("voiceSelect");
  const autoSpeak = document.getElementById("autoSpeak");
  const details = document.getElementById("details");

  let conversationId = null;
  let proposalId = null;
  let latestAgentMessage = "";
  let recognition = null;
  let finalTranscript = "";
  let requestInFlight = false;

  function addBubble(role, text) {
    const bubble = document.createElement("div");
    bubble.className = "bubble " + (role === "caller" ? "caller" : "agent");

    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = role === "caller" ? "You" : "Agent";

    const content = document.createElement("div");
    content.textContent = text;

    bubble.append(speaker, content);
    chatLog.appendChild(bubble);
    bubble.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function populateVoices() {
    if (!("speechSynthesis" in window)) {
      voiceSelect.disabled = true;
      voiceSelect.replaceChildren(new Option("Speech unavailable", ""));
      return;
    }

    const previous = voiceSelect.value;
    const voices = window.speechSynthesis.getVoices()
      .filter((voice) => /^en[-_]/i.test(voice.lang));

    const rank = (voice) => {
      if (/^en[-_]IN$/i.test(voice.lang)) return 0;
      if (/^en[-_]GB$/i.test(voice.lang)) return 1;
      if (/^en[-_]US$/i.test(voice.lang)) return 2;
      return 3;
    };

    voices.sort((a, b) => rank(a) - rank(b) ||
      a.name.localeCompare(b.name));

    voiceSelect.replaceChildren();
    for (const voice of voices) {
      voiceSelect.add(new Option(
        `${voice.name} (${voice.lang})`, voice.voiceURI
      ));
    }
    if (!voices.length) {
      voiceSelect.add(new Option("Browser default voice", ""));
    } else if (voices.some((voice) => voice.voiceURI === previous)) {
      voiceSelect.value = previous;
    } else {
      voiceSelect.value = voices[0].voiceURI;
    }
  }

  function speak(text) {
    if (!("speechSynthesis" in window) || !text) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = window.speechSynthesis.getVoices().find(
      (candidate) => candidate.voiceURI === voiceSelect.value
    );
    if (voice) utterance.voice = voice;
    utterance.rate = 0.94;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }

  function showAgentResponse(data) {
    const message = data.agent_message || data.message ||
      "I couldn't complete that step. No ticket success is being claimed.";

    latestAgentMessage = message;
    addBubble("agent", message);

    if (data.conversation_id) conversationId = data.conversation_id;
    proposalId = data.proposal_id || null;
    confirmButton.hidden = !proposalId;

    details.textContent = JSON.stringify({
      status: data.status || null,
      revision: data.revision ?? null,
      extraction: data.extraction || null,
      proposal: data.proposal || null,
      workflow_data: data.data || null
    }, null, 2);

    if (autoSpeak.checked) speak(message);
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "The request failed.");
    }
    return data;
  }

  async function sendTurn(rawText) {
    const text = (rawText || reportInput.value).trim();
    if (!text || requestInFlight) return;

    requestInFlight = true;
    sendButton.disabled = true;
    addBubble("caller", text);
    reportInput.value = "";
    voiceStatus.textContent = "Thinking through your message...";

    try {
      const result = await postJson("/api/turn", {
        conversation_id: conversationId,
        text
      });
      showAgentResponse(result);
      voiceStatus.textContent =
        "Your turn is complete. Speak again or type a correction or reply.";
    } catch (error) {
      const message = error.message ||
        "The request failed. No ticket success is being claimed.";
      latestAgentMessage = message;
      addBubble("agent", message);
      if (autoSpeak.checked) speak(message);
      voiceStatus.textContent = "Request failed. You can try again.";
    } finally {
      requestInFlight = false;
      sendButton.disabled = false;
      reportInput.focus();
    }
  }

  sendButton.addEventListener("click", () => sendTurn(reportInput.value));

  reportInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      sendTurn(reportInput.value);
    }
  });

  confirmButton.addEventListener("click", async () => {
    if (!proposalId || requestInFlight) return;

    requestInFlight = true;
    confirmButton.disabled = true;
    try {
      const result = await postJson(
        "/api/proposals/" + encodeURIComponent(proposalId) + "/confirm",
        { confirmed: true }
      );
      showAgentResponse(result);
    } catch (error) {
      const message = error.message || "Confirmation failed.";
      latestAgentMessage = message;
      addBubble("agent", message);
      if (autoSpeak.checked) speak(message);
    } finally {
      requestInFlight = false;
      confirmButton.disabled = false;
    }
  });

  document.getElementById("repeatResponse").addEventListener("click", () => {
    speak(latestAgentMessage);
  });

  if ("speechSynthesis" in window) {
    populateVoices();
    window.speechSynthesis.addEventListener("voiceschanged", populateVoices);
  } else {
    populateVoices();
  }

  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    document.getElementById("startVoice").disabled = true;
    voiceStatus.textContent =
      "Voice recognition is unavailable in this browser. You can type replies.";
  } else {
    recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      finalTranscript = "";
      voiceStatus.textContent =
        "Listening. When you finish speaking, your turn will be sent.";
      document.getElementById("startVoice").disabled = true;
      document.getElementById("stopVoice").disabled = false;
    };

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript + " ";
        }
      }
    };

    recognition.onerror = (event) => {
      if (event.error === "not-allowed" ||
          event.error === "service-not-allowed") {
        voiceStatus.textContent =
          "Microphone access was blocked. Allow microphone access for " +
          "127.0.0.1 in Chrome site settings, check Windows microphone " +
          "privacy settings, then reload. You can still type replies.";
      } else {
        voiceStatus.textContent =
          "Speech recognition error: " + event.error +
          ". You can still type replies.";
      }
    };

    recognition.onend = () => {
      document.getElementById("startVoice").disabled = false;
      document.getElementById("stopVoice").disabled = true;

      const text = finalTranscript.trim();
      finalTranscript = "";
      if (text) {
        sendTurn(text);
      } else {
        voiceStatus.textContent = "No speech captured. Try again or type a reply.";
      }
    };

    document.getElementById("startVoice").addEventListener("click", () => {
      try {
        recognition.start();
      } catch (_) {
        voiceStatus.textContent =
          "Voice capture could not start. Try again or type your reply.";
      }
    });

    document.getElementById("stopVoice").addEventListener("click", () => {
      recognition.stop();
    });
  }
})();
</script>
</body>
</html>"""


def create_app(
    *,
    runtime: IncidentRuntime | None = None,
    database_path: str | Path = DEFAULT_DATABASE,
) -> FastAPI:
    """Build a local-only app; allow dependency injection for tests."""
    application = FastAPI(
        title="Field Incident Intake Demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    runtime_holder = {"value": runtime}
    pending: dict[str, tuple[object, AccessContext, str]] = {}
    conversation_manager_holder: dict[str, ConversationManager | None] = {"value": None}

    def get_runtime() -> IncidentRuntime:
        if runtime_holder["value"] is None:
            try:
                runtime_holder["value"] = build_runtime(database_path)
            except ModelClientError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Model unavailable. Check GROQ_API_KEY and retry.",
                ) from exc

        active_runtime = runtime_holder["value"]
        _ensure_demo_assets(active_runtime)
        return active_runtime

    def get_conversation_manager() -> ConversationManager:
        active_runtime = get_runtime()
        if conversation_manager_holder["value"] is None:
            conversation_manager_holder["value"] = ConversationManager(
                active_runtime
            )
        return conversation_manager_holder["value"]

    @application.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @application.post("/api/turn")
    def conversational_turn(request: TurnRequest) -> dict[str, object]:
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="Message cannot be blank.")

        try:
            return get_conversation_manager().turn(
                request.conversation_id,
                request.text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except IncidentValidationError as exc:
            raise HTTPException(
                status_code=409,
                detail="The intake could not advance safely.",
            ) from exc

    @application.post("/api/intake")
    def intake(request: IntakeRequest) -> dict[str, object]:
        if not request.text.strip():
            raise HTTPException(
                status_code=422,
                detail="Report text cannot be blank.",
            )

        active_runtime = get_runtime()
        access = AccessContext(
            principal_id=DEMO_PRINCIPAL_ID,
            authorized_site_ids=frozenset({DEMO_SITE_ID}),
        )
        session_id = f"SESSION-WEB-{uuid.uuid4().hex}"
        idempotency_key = f"IDEM-WEB-{uuid.uuid4().hex}"

        try:
            active_runtime.workflow.start(session_id, access)
            outcome = active_runtime.orchestrator.process_report(
                session_id,
                request.text,
                access,
                idempotency_key,
            )
        except IncidentValidationError as exc:
            raise HTTPException(
                status_code=409,
                detail="The intake could not advance safely.",
            ) from exc

        response: dict[str, object] = {
            "status": outcome.reply.status,
            "message": outcome.reply.message,
            "data": outcome.reply.data,
            "extraction": (
                asdict(outcome.extraction)
                if outcome.extraction is not None
                else None
            ),
        }

        if outcome.reply.status == "confirmation_required":
            proposal_id = secrets.token_urlsafe(32)
            pending[proposal_id] = (outcome, access, idempotency_key)

            while len(pending) > MAX_PENDING_PROPOSALS:
                pending.pop(next(iter(pending)))

            response["proposal_id"] = proposal_id
            response["proposal"] = outcome.reply.data.get("proposal")

        return response

    @application.post("/api/proposals/{proposal_id}/confirm")
    def confirm(
        proposal_id: str,
        request: ConfirmationRequest,
    ) -> dict[str, object]:
        if request.confirmed is not True:
            raise HTTPException(
                status_code=400,
                detail="Explicit confirmation is required.",
            )

        manager = conversation_manager_holder["value"]
        if manager is not None and manager.has_proposal(proposal_id):
            try:
                return manager.confirm_by_proposal(proposal_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="Proposal not found or no longer available.",
                ) from exc

        entry = pending.get(proposal_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="Proposal not found or no longer available.",
            )

        outcome, access, idempotency_key = entry
        active_runtime = get_runtime()

        try:
            reply = active_runtime.orchestrator.confirm_proposal(
                outcome,
                access,
                idempotency_key,
                explicitly_confirmed=True,
            )
        except IncidentValidationError as exc:
            raise HTTPException(
                status_code=409,
                detail="The proposal could not be confirmed safely.",
            ) from exc

        return {
            "status": reply.status,
            "message": reply.message,
            "data": reply.data,
        }

    return application


app = create_app()