"use client";

// =============================================================================
// page.tsx — Aegis RAG · Medical Intelligence Hub
// Full dashboard: sidebar (PDF drop-zone + report list) + chat window with
// citation accordions, low-confidence warnings, and medical disclaimer.
// All state is local (useState); zero console logging; pure TypeScript.
// =============================================================================

import React, { useState, useRef, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type Citation = {
  chunkId: string;
  page: number;
  source: string;
  snippet: string;
};

type Message = {
  id: string;
  role: "user" | "ai";
  content: string;
  citations?: Citation[];
  lowConfidence: boolean;
  timestamp: Date;
};

type Report = {
  id: string;
  name: string;
  pages: number;
  chunks: number;
  size: string;
  uploadedAt: string;
};

type MockReply = {
  content: string;
  citations?: Citation[];
  lowConfidence: boolean;
};

// ─────────────────────────────────────────────────────────────────────────────
// Seed data — pre-loaded mock reports
// ─────────────────────────────────────────────────────────────────────────────

const SEED_REPORTS: Report[] = [
  {
    id: "r1",
    name: "CardiacAssessment_2024.pdf",
    pages: 12,
    chunks: 47,
    size: "2.4 MB",
    uploadedAt: "Jul 14, 2026",
  },
  {
    id: "r2",
    name: "LabResults_HbA1c_Q2.pdf",
    pages: 4,
    chunks: 16,
    size: "890 KB",
    uploadedAt: "Jul 16, 2026",
  },
  {
    id: "r3",
    name: "RadiologyReport_ChestXR.pdf",
    pages: 6,
    chunks: 23,
    size: "5.1 MB",
    uploadedAt: "Jul 17, 2026",
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Seed data — initial chat history
// ─────────────────────────────────────────────────────────────────────────────

const SEED_MESSAGES: Message[] = [
  {
    id: "seed-1",
    role: "user",
    content: "What were the key findings in the cardiac assessment?",
    lowConfidence: false,
    timestamp: new Date(Date.now() - 5 * 60_000),
  },
  {
    id: "seed-2",
    role: "ai",
    content:
      "The cardiac assessment revealed a mildly elevated LDL cholesterol level of 148 mg/dL [chunk_id: cardiac_p3_c2, page: 3]. The ejection fraction was measured at 58%, within the normal range [chunk_id: cardiac_p5_c1, page: 5]. No significant ST-segment changes were noted on the resting ECG [chunk_id: cardiac_p7_c3, page: 7].",
    citations: [
      {
        chunkId: "cardiac_p3_c2",
        page: 3,
        source: "CardiacAssessment_2024.pdf",
        snippet:
          "Serum LDL cholesterol: 148 mg/dL (reference range < 100 mg/dL optimal, < 130 mg/dL near optimal). Patient counselled on dietary modifications and increased aerobic activity.",
      },
      {
        chunkId: "cardiac_p5_c1",
        page: 5,
        source: "CardiacAssessment_2024.pdf",
        snippet:
          "Echocardiographic assessment: Left ventricular ejection fraction (LVEF) = 58%. No regional wall motion abnormalities detected. Mild diastolic dysfunction grade I.",
      },
      {
        chunkId: "cardiac_p7_c3",
        page: 7,
        source: "CardiacAssessment_2024.pdf",
        snippet:
          "12-lead resting ECG: Normal sinus rhythm at 72 bpm. No ST-segment elevation or depression. No T-wave inversions. QTc interval 420 ms.",
      },
    ],
    lowConfidence: false,
    timestamp: new Date(Date.now() - 4 * 60_000),
  },
  {
    id: "seed-3",
    role: "user",
    content: "What is the patient's HbA1c level and clinical target?",
    lowConfidence: false,
    timestamp: new Date(Date.now() - 2 * 60_000),
  },
  {
    id: "seed-4",
    role: "ai",
    content:
      "The patient's HbA1c was recorded at 7.2% in Q2 2026 [chunk_id: hba1c_p2_c1, page: 2], indicating suboptimal glycaemic control. The clinical target for most adults with type 2 diabetes is below 7.0% per ADA 2024 standards [chunk_id: hba1c_p3_c2, page: 3].",
    citations: [
      {
        chunkId: "hba1c_p2_c1",
        page: 2,
        source: "LabResults_HbA1c_Q2.pdf",
        snippet:
          "HbA1c (Glycated Haemoglobin): 7.2% (55 mmol/mol). Result date: 10-Jun-2026. Previous value: 7.8% (62 mmol/mol) — improvement over the 3-month interval.",
      },
      {
        chunkId: "hba1c_p3_c2",
        page: 3,
        source: "LabResults_HbA1c_Q2.pdf",
        snippet:
          "Clinical target: HbA1c < 7.0% (53 mmol/mol) per ADA 2024 Standards of Care for adults with T2DM without significant comorbidities.",
      },
    ],
    lowConfidence: false,
    timestamp: new Date(Date.now() - 1 * 60_000),
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Mock AI reply pool — cycled on every user send
// ─────────────────────────────────────────────────────────────────────────────

const AI_REPLY_POOL: MockReply[] = [
  {
    content:
      "The chest X-ray demonstrated mild cardiomegaly with a cardiothoracic ratio of 0.52 [chunk_id: xr_p2_c1, page: 2]. Both lung fields were clear with no consolidation, pleural effusion, or pneumothorax identified [chunk_id: xr_p3_c2, page: 3].",
    citations: [
      {
        chunkId: "xr_p2_c1",
        page: 2,
        source: "RadiologyReport_ChestXR.pdf",
        snippet:
          "Cardiothoracic (CT) ratio: 0.52. Mild cardiomegaly noted. Right heart border well-defined. Aortic knuckle prominent. Clinical correlation recommended.",
      },
      {
        chunkId: "xr_p3_c2",
        page: 3,
        source: "RadiologyReport_ChestXR.pdf",
        snippet:
          "Lung fields: Clear bilaterally. No focal consolidation. No pleural effusion. No evidence of pneumothorax. Vascular markings within normal limits.",
      },
    ],
    lowConfidence: false,
  },
  {
    content: "I cannot find the answer in the provided document.",
    lowConfidence: true,
  },
  {
    content:
      "The fasting blood glucose was 118 mg/dL [chunk_id: lab_p1_c3, page: 1], classifying as impaired fasting glucose per WHO criteria. The treating clinician recommended a repeat OGTT in 3 months alongside lifestyle intervention [chunk_id: lab_p2_c1, page: 2].",
    citations: [
      {
        chunkId: "lab_p1_c3",
        page: 1,
        source: "LabResults_HbA1c_Q2.pdf",
        snippet:
          "Fasting plasma glucose: 118 mg/dL (6.6 mmol/L). Classification: Impaired Fasting Glucose (IFG) per WHO criteria (≥ 100 and < 126 mg/dL).",
      },
      {
        chunkId: "lab_p2_c1",
        page: 2,
        source: "LabResults_HbA1c_Q2.pdf",
        snippet:
          "Recommendation: Repeat oral glucose tolerance test (OGTT) in 3 months. Lifestyle intervention counselling initiated. Dietary referral placed.",
      },
    ],
    lowConfidence: false,
  },
  {
    content:
      "Current medications include Metformin 500 mg PO twice daily [chunk_id: cardiac_p9_c1, page: 9] and Atorvastatin 20 mg PO once nightly [chunk_id: cardiac_p9_c2, page: 9]. A follow-up was scheduled 6 weeks post-assessment [chunk_id: cardiac_p11_c1, page: 11].",
    citations: [
      {
        chunkId: "cardiac_p9_c1",
        page: 9,
        source: "CardiacAssessment_2024.pdf",
        snippet:
          "Metformin 500 mg PO BID (twice daily) — for glycaemic management in T2DM. Patient reports good tolerability. No GI adverse effects documented.",
      },
      {
        chunkId: "cardiac_p9_c2",
        page: 9,
        source: "CardiacAssessment_2024.pdf",
        snippet:
          "Atorvastatin 20 mg PO QHS (once at bedtime) — initiated for hyperlipidaemia. LFTs to be reviewed at next visit. Patient advised on myopathy symptoms.",
      },
    ],
    lowConfidence: false,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

// Stable formatter — uses fixed "HH:MM" format to avoid SSR/client mismatch.
function formatTime(date: Date): string {
  const h = date.getHours().toString().padStart(2, "0");
  const m = date.getMinutes().toString().padStart(2, "0");
  return `${h}:${m}`;
}

// Renders children only after the first client-side paint to avoid hydration
// mismatches for any time/date values that differ between server and browser.
function ClientOnly({ children }: { children: React.ReactNode }): React.JSX.Element | null {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;
  return <>{children}</>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Animated ECG pulse icon
// ─────────────────────────────────────────────────────────────────────────────

function EcgPulseIcon(): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 80 28"
      className="w-20 h-7"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ animation: "ecgDraw 2.4s ease-in-out infinite" }}
    >
      <polyline points="0,14 12,14 17,4 22,24 27,14 36,14 41,2 46,26 51,14 60,14 65,9 70,19 75,14 80,14" />
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Report card in sidebar
// ─────────────────────────────────────────────────────────────────────────────

function ReportCard({
  report,
  isActive,
  onClick,
}: {
  report: Report;
  isActive: boolean;
  onClick: () => void;
}): React.JSX.Element {
  return (
    <button
      type="button"
      id={`report-${report.id}`}
      onClick={onClick}
      className={[
        "w-full text-left rounded-xl px-4 py-3 border transition-all duration-200",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-oceanic-blue",
        isActive
          ? "bg-oceanic-blue border-oceanic-blue shadow-md"
          : "bg-calming-neutral-card border-calming-neutral-border hover:border-oceanic-blue hover:shadow-sm",
      ].join(" ")}
    >
      <div className="flex items-start gap-3">
        <span className="text-xl mt-0.5 shrink-0" aria-hidden="true">
          📄
        </span>
        <div className="min-w-0 flex-1">
          <p
            className={`text-sm font-semibold truncate ${
              isActive ? "text-white" : "text-calming-neutral-text"
            }`}
          >
            {report.name}
          </p>
          <p
            className={`text-xs mt-0.5 ${isActive ? "text-blue-100" : "text-calming-neutral-text"}`}
            style={{ opacity: isActive ? 0.75 : 0.55 }}
          >
            {report.pages} pages · {report.chunks} chunks · {report.size}
          </p>
          <p
            className={`text-xs mt-0.5 ${isActive ? "text-blue-100" : "text-calming-neutral-text"}`}
            style={{ opacity: isActive ? 0.55 : 0.38 }}
          >
            Uploaded {report.uploadedAt}
          </p>
        </div>
      </div>
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Citation accordion (details / summary)
// ─────────────────────────────────────────────────────────────────────────────

function CitationAccordion({
  citations,
}: {
  citations: Citation[];
}): React.JSX.Element {
  return (
    <div className="mt-3 space-y-1.5">
      <p className="text-[10px] font-bold text-healing-green-dark uppercase tracking-widest mb-2">
        📎 Source Evidence ({citations.length})
      </p>
      {citations.map((c) => (
        <details
          key={c.chunkId}
          className="group rounded-lg border border-calming-neutral-border bg-calming-neutral-canvas overflow-hidden"
        >
          <summary className="flex items-center justify-between gap-2 px-3 py-2 cursor-pointer text-xs font-medium text-calming-neutral-text select-none list-none hover:bg-calming-neutral-border transition-colors duration-150">
            <span className="min-w-0 truncate">
              <span className="font-mono text-oceanic-blue">{c.chunkId}</span>
              <span
                className="text-calming-neutral-text ml-1.5"
                style={{ opacity: 0.45 }}
              >
                pg.{c.page} · {c.source}
              </span>
            </span>
            <svg
              className="w-3.5 h-3.5 shrink-0 text-calming-neutral-text transition-transform duration-200 group-open:rotate-180"
              style={{ opacity: 0.4 }}
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </summary>
          <div className="px-3 pb-3 pt-2 border-t border-calming-neutral-border">
            <blockquote
              className="text-xs leading-relaxed text-calming-neutral-text italic pl-3 border-l-2 border-oceanic-blue"
              style={{ opacity: 0.75 }}
            >
              {c.snippet}
            </blockquote>
          </div>
        </details>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Single message bubble
// ─────────────────────────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: Message }): React.JSX.Element {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%]">
          <div className="bg-oceanic-blue text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm">
            <p className="text-sm leading-relaxed">{message.content}</p>
          </div>
          <p
            className="text-right text-[10px] text-calming-neutral-text mt-1 pr-1"
            style={{ opacity: 0.38 }}
          >
            {formatTime(message.timestamp)}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[82%]">
        {/* Avatar row */}
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-6 h-6 rounded-full bg-healing-green flex items-center justify-center text-white text-[10px] font-bold shrink-0">
            AI
          </div>
          <span className="text-xs font-semibold text-healing-green-dark">
            Aegis AI
          </span>
          <span
            className="text-[10px] text-calming-neutral-text"
            style={{ opacity: 0.38 }}
          >
            {formatTime(message.timestamp)}
          </span>
        </div>

        {/* Bubble */}
        <div
          className={[
            "rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border",
            message.lowConfidence
              ? "bg-vitality-red-light border-vitality-red"
              : "bg-healing-green-light border-healing-green",
          ].join(" ")}
          style={{ borderOpacity: 0.25 }}
        >
          {/* Low-confidence inline badge */}
          {message.lowConfidence && (
            <div
              className="flex items-center gap-1.5 mb-2 pb-2 border-b border-vitality-red"
              style={{ borderOpacity: 0.25 }}
            >
              <span className="text-sm" aria-hidden="true">
                ⚠️
              </span>
              <span className="text-xs font-bold text-vitality-red-dark">
                Low Confidence
              </span>
            </div>
          )}

          <p className="text-sm leading-relaxed text-calming-neutral-text">
            {message.content}
          </p>

          {message.citations && message.citations.length > 0 && (
            <CitationAccordion citations={message.citations} />
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Typing indicator (three bouncing dots)
// ─────────────────────────────────────────────────────────────────────────────

function TypingIndicator(): React.JSX.Element {
  return (
    <div className="flex justify-start items-end gap-2">
      <div className="w-6 h-6 rounded-full bg-healing-green flex items-center justify-center text-white text-[10px] font-bold shrink-0">
        AI
      </div>
      <div
        className="bg-healing-green-light border border-healing-green rounded-2xl rounded-tl-sm px-4 py-3"
        style={{ borderOpacity: 0.25 }}
      >
        <div className="flex gap-1.5 items-center h-4">
          {([0, 150, 300] as const).map((delay) => (
            <span
              key={delay}
              className="w-2 h-2 bg-healing-green rounded-full animate-bounce"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Low-confidence global alert banner
// ─────────────────────────────────────────────────────────────────────────────

function LowConfidenceAlert(): React.JSX.Element {
  return (
    <div
      className="mx-4 mb-2 flex items-start gap-3 rounded-xl bg-vitality-red-light border border-vitality-red px-4 py-3"
      role="alert"
      aria-live="polite"
      style={{ animation: "fadeSlideIn 0.3s ease-out", borderOpacity: 0.35 }}
    >
      <span className="text-xl mt-0.5 shrink-0" aria-hidden="true">
        🚨
      </span>
      <div>
        <p className="text-sm font-bold text-vitality-red-dark">
          Low Confidence Warning
        </p>
        <p
          className="text-xs text-vitality-red-dark mt-0.5 leading-relaxed"
          style={{ opacity: 0.8 }}
        >
          The AI could not reliably locate an answer in the indexed documents.
          The retrieved context may be insufficient or unrelated to your query.
          Please verify with a qualified healthcare professional before acting on
          this information.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Permanent medical disclaimer bar
// ─────────────────────────────────────────────────────────────────────────────

function MedicalDisclaimer(): React.JSX.Element {
  return (
    <div
      className="px-4 py-2.5 flex items-start gap-2 bg-gentle-yellow-light border-t border-gentle-yellow"
      role="note"
      aria-label="Medical disclaimer"
      style={{ borderOpacity: 0.5 }}
    >
      <span className="text-base mt-0.5 shrink-0" aria-hidden="true">
        ⚕️
      </span>
      <p className="text-[10.5px] leading-relaxed text-gentle-yellow-dark">
        <strong>Medical Disclaimer:</strong> This AI tool is for informational
        and educational purposes only. It does not constitute professional
        medical advice, diagnosis, or treatment. Always consult a qualified
        healthcare professional for any medical decisions.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page component (default export)
// ─────────────────────────────────────────────────────────────────────────────

export default function Home(): React.JSX.Element {
  // ── State ──────────────────────────────────────────────────────────────────
  const [reports] = useState<Report[]>(SEED_REPORTS);
  const [activeReportId, setActiveReportId] = useState<string>("r1");
  const [messages, setMessages] = useState<Message[]>(SEED_MESSAGES);
  const [inputValue, setInputValue] = useState<string>("");
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [showLowConfidenceAlert, setShowLowConfidenceAlert] =
    useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [replyIndex, setReplyIndex] = useState<number>(0);

  // ── Refs ───────────────────────────────────────────────────────────────────
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // ── Auto-scroll to bottom on new messages ─────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // ── Send handler ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isTyping) return;

    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: text,
      lowConfidence: false,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsTyping(true);
    setShowLowConfidenceAlert(false);

    // Simulate network / inference latency
    await new Promise<void>((resolve) =>
      setTimeout(resolve, 1200 + Math.random() * 900)
    );

    const reply = AI_REPLY_POOL[replyIndex % AI_REPLY_POOL.length];
    setReplyIndex((i) => i + 1);

    const aiMsg: Message = {
      id: uid(),
      role: "ai",
      content: reply.content,
      citations: reply.citations,
      lowConfidence: reply.lowConfidence,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, aiMsg]);
    setIsTyping(false);

    if (reply.lowConfidence) {
      setShowLowConfidenceAlert(true);
    }
  }, [inputValue, isTyping, replyIndex]);

  // ── Keyboard shortcut: Enter to send, Shift+Enter for newline ─────────────
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ── Drag-and-drop handlers (UI feedback only) ─────────────────────────────
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <>
      {/* ── Global keyframe animations ─────────────────────────────────────── */}
      <style>{`
        @keyframes ecgDraw {
          0%   { stroke-dasharray: 300; stroke-dashoffset: 300; opacity: 0.5; }
          40%  { opacity: 1; }
          80%  { stroke-dashoffset: 0; }
          100% { stroke-dashoffset: 0; opacity: 0.5; }
        }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(-8px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
        @keyframes popIn {
          from { opacity: 0; transform: scale(0.96) translateY(6px); }
          to   { opacity: 1; transform: scale(1)    translateY(0);   }
        }
        .msg-in { animation: popIn 0.22s ease-out; }
        details summary::-webkit-details-marker { display: none; }
      `}</style>

      <div className="flex flex-col min-h-screen bg-calming-neutral-canvas">

        {/* ══════════════════════════════════════════════════════════════════
            HEADER
        ══════════════════════════════════════════════════════════════════ */}
        <header
          className="sticky top-0 z-30 bg-calming-neutral-card border-b border-calming-neutral-border shadow-sm"
          role="banner"
        >
          <div className="max-w-screen-xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">

            {/* Branding */}
            <div className="flex items-center gap-3">
              <div className="text-vitality-red shrink-0">
                <EcgPulseIcon />
              </div>
              <div>
                <h1 className="text-base font-bold text-oceanic-blue leading-tight tracking-tight">
                  Aegis RAG
                </h1>
                <p
                  className="text-[11px] text-calming-neutral-text leading-tight"
                  style={{ opacity: 0.5 }}
                >
                  Medical Intelligence Hub
                </p>
              </div>
            </div>

            {/* Status pills */}
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-healing-green-light px-3 py-1 text-xs font-semibold text-healing-green-dark">
                <span className="w-1.5 h-1.5 rounded-full bg-healing-green animate-pulse" />
                RAG Online
              </span>
              <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-oceanic-blue-light px-3 py-1 text-xs font-semibold text-oceanic-blue-dark">
                llama-3.3-70b · Groq
              </span>
            </div>
          </div>
        </header>

        {/* ══════════════════════════════════════════════════════════════════
            BODY (sidebar + chat panel)
        ══════════════════════════════════════════════════════════════════ */}
        <div className="flex-1 max-w-screen-xl mx-auto w-full px-4 sm:px-6 py-5 flex gap-5 min-h-0"
          style={{ height: "calc(100vh - 4rem)" }}
        >

          {/* ── LEFT SIDEBAR ─────────────────────────────────────────────── */}
          <aside
            className="hidden lg:flex flex-col w-72 shrink-0 gap-4 overflow-y-auto"
            aria-label="Document sidebar"
          >

            {/* PDF Drop-zone */}
            <div
              id="pdf-dropzone"
              role="region"
              aria-label="PDF upload drop zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={[
                "rounded-2xl border-2 border-dashed p-6 flex flex-col items-center gap-2 text-center cursor-pointer transition-all duration-200 shrink-0",
                isDragging
                  ? "border-oceanic-blue bg-oceanic-blue-light scale-[1.02]"
                  : "border-calming-neutral-border bg-calming-neutral-card hover:border-oceanic-blue",
              ].join(" ")}
            >
              <span
                className={`text-4xl transition-transform duration-200 ${isDragging ? "scale-125" : ""}`}
                aria-hidden="true"
              >
                📂
              </span>
              <p className="text-sm font-semibold text-calming-neutral-text">
                Drop PDF reports here
              </p>
              <p
                className="text-xs text-calming-neutral-text"
                style={{ opacity: 0.5 }}
              >
                or{" "}
                <label className="text-oceanic-blue underline cursor-pointer hover:text-oceanic-blue-dark transition-colors">
                  browse files
                  <input
                    type="file"
                    id="file-upload-input"
                    accept=".pdf"
                    multiple
                    className="sr-only"
                  />
                </label>
              </p>
              <p
                className="text-[10px] text-calming-neutral-text mt-1"
                style={{ opacity: 0.3 }}
              >
                Max 50 MB per file · PDF only
              </p>
            </div>

            {/* Indexed report list */}
            <div className="flex-1 rounded-2xl bg-calming-neutral-card border border-calming-neutral-border p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between shrink-0">
                <h2
                  className="text-[10px] font-bold text-calming-neutral-text uppercase tracking-widest"
                  style={{ opacity: 0.6 }}
                >
                  Indexed Reports
                </h2>
                <span className="text-xs bg-oceanic-blue-light text-oceanic-blue-dark font-bold px-2 py-0.5 rounded-full">
                  {reports.length}
                </span>
              </div>
              <div className="flex flex-col gap-2">
                {reports.map((r) => (
                  <ReportCard
                    key={r.id}
                    report={r}
                    isActive={activeReportId === r.id}
                    onClick={() => setActiveReportId(r.id)}
                  />
                ))}
              </div>
            </div>

            {/* Stats grid */}
            <div className="rounded-2xl bg-calming-neutral-card border border-calming-neutral-border p-4 grid grid-cols-2 gap-3 shrink-0">
              <div className="text-center p-2 rounded-xl bg-oceanic-blue-light">
                <p className="text-lg font-bold text-oceanic-blue">86</p>
                <p
                  className="text-[9px] text-oceanic-blue-dark font-semibold uppercase tracking-wider"
                  style={{ opacity: 0.7 }}
                >
                  Chunks
                </p>
              </div>
              <div className="text-center p-2 rounded-xl bg-healing-green-light">
                <p className="text-lg font-bold text-healing-green-dark">22</p>
                <p
                  className="text-[9px] text-healing-green-dark font-semibold uppercase tracking-wider"
                  style={{ opacity: 0.7 }}
                >
                  Pages
                </p>
              </div>
              <div className="text-center p-2 rounded-xl bg-gentle-yellow-light">
                <p className="text-lg font-bold text-gentle-yellow-dark">3</p>
                <p
                  className="text-[9px] text-gentle-yellow-dark font-semibold uppercase tracking-wider"
                  style={{ opacity: 0.7 }}
                >
                  Docs
                </p>
              </div>
              <div className="text-center p-2 rounded-xl bg-vitality-red-light">
                <p className="text-lg font-bold text-vitality-red-dark">
                  8.4<span className="text-xs">MB</span>
                </p>
                <p
                  className="text-[9px] text-vitality-red-dark font-semibold uppercase tracking-wider"
                  style={{ opacity: 0.7 }}
                >
                  Size
                </p>
              </div>
            </div>
          </aside>

          {/* ── MAIN CHAT PANEL ──────────────────────────────────────────── */}
          <main
            id="chat-panel"
            className="flex-1 flex flex-col bg-calming-neutral-card rounded-2xl border border-calming-neutral-border shadow-sm overflow-hidden"
            aria-label="Chat interface"
          >
            {/* Chat top bar */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-calming-neutral-border bg-calming-neutral-canvas shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-full bg-healing-green flex items-center justify-center text-white text-xs font-bold shrink-0">
                  AI
                </div>
                <div>
                  <p className="text-sm font-bold text-calming-neutral-text">
                    Aegis AI
                  </p>
                  <p
                    className="text-[10px] text-calming-neutral-text"
                    style={{ opacity: 0.45 }}
                  >
                    llama-3.3-70b-versatile · Groq Inference API
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-healing-green animate-pulse" />
                <span className="text-xs font-semibold text-healing-green-dark">
                  Active
                </span>
              </div>
            </div>

            {/* Messages scroll area */}
            <div
              id="messages-area"
              className="flex-1 overflow-y-auto px-5 py-5 space-y-4 min-h-0"
              aria-live="polite"
              aria-relevant="additions"
            >
              {messages.map((msg) => (
                <div key={msg.id} className="msg-in">
                  <MessageBubble message={msg} />
                </div>
              ))}

              {isTyping && <TypingIndicator />}

              {/* Bottom scroll anchor */}
              <div ref={messagesEndRef} />
            </div>

            {/* Low confidence global alert */}
            {showLowConfidenceAlert && <LowConfidenceAlert />}

            {/* Input + disclaimer area */}
            <div className="shrink-0 border-t border-calming-neutral-border">
              <div className="px-4 pt-3 pb-1 flex gap-2 items-end">
                <textarea
                  ref={inputRef}
                  id="chat-input"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a clinical question about your documents…"
                  rows={2}
                  disabled={isTyping}
                  aria-label="Chat input"
                  className="flex-1 resize-none rounded-xl border border-calming-neutral-border bg-calming-neutral-canvas px-4 py-3 text-sm text-calming-neutral-text focus:outline-none focus:ring-2 focus:ring-oceanic-blue focus:border-transparent transition-all duration-150 disabled:opacity-50"
                />

                {/* Send button */}
                <button
                  id="send-button"
                  type="button"
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isTyping}
                  aria-label="Send message"
                  className="shrink-0 w-11 h-11 rounded-xl bg-oceanic-blue text-white flex items-center justify-center hover:bg-oceanic-blue-dark active:scale-95 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-oceanic-blue focus-visible:ring-offset-2"
                >
                  <svg
                    viewBox="0 0 24 24"
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M22 2L11 13" />
                    <path d="M22 2L15 22l-4-9-9-4 20-7z" />
                  </svg>
                </button>
              </div>

              {/* Keyboard hint */}
              <div className="px-4 pb-2 flex items-center gap-1">
                <kbd
                  className="text-[10px] text-calming-neutral-text bg-calming-neutral-canvas border border-calming-neutral-border rounded px-1.5 py-0.5 font-mono"
                  style={{ opacity: 0.45 }}
                >
                  Enter
                </kbd>
                <span
                  className="text-[10px] text-calming-neutral-text"
                  style={{ opacity: 0.35 }}
                >
                  to send ·
                </span>
                <kbd
                  className="text-[10px] text-calming-neutral-text bg-calming-neutral-canvas border border-calming-neutral-border rounded px-1.5 py-0.5 font-mono"
                  style={{ opacity: 0.45 }}
                >
                  Shift+Enter
                </kbd>
                <span
                  className="text-[10px] text-calming-neutral-text"
                  style={{ opacity: 0.35 }}
                >
                  for new line
                </span>
              </div>

              {/* Permanent medical disclaimer */}
              <MedicalDisclaimer />
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
