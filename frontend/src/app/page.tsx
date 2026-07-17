"use client";

// =============================================================================
// page.tsx — Medical Intelligence Hub
// Premium card-based dashboard: soft mint/aqua canvas, translucent white
// cards, polished report list, citation reference cards, and integrated safety
// guardrails. All state is local; zero console logging; pure TypeScript.
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
  status: "indexed" | "processing";
};

type MockReply = {
  content: string;
  citations?: Citation[];
  lowConfidence: boolean;
};

// ─────────────────────────────────────────────────────────────────────────────
// Seed data
// ─────────────────────────────────────────────────────────────────────────────

const SEED_REPORTS: Report[] = [
  {
    id: "r1",
    name: "CardiacAssessment_2024.pdf",
    pages: 12,
    chunks: 47,
    size: "2.4 MB",
    uploadedAt: "Jul 14, 2026",
    status: "indexed",
  },
  {
    id: "r2",
    name: "LabResults_HbA1c_Q2.pdf",
    pages: 4,
    chunks: 16,
    size: "890 KB",
    uploadedAt: "Jul 16, 2026",
    status: "indexed",
  },
  {
    id: "r3",
    name: "RadiologyReport_ChestXR.pdf",
    pages: 6,
    chunks: 23,
    size: "5.1 MB",
    uploadedAt: "Jul 17, 2026",
    status: "indexed",
  },
];

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
      "The cardiac assessment revealed a mildly elevated LDL cholesterol level of 148 mg/dL. The ejection fraction was measured at 58%, within the normal range. No significant ST-segment changes were noted on the resting ECG.",
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
      "The patient's HbA1c was recorded at 7.2% in Q2 2026, indicating suboptimal glycaemic control. The clinical target for most adults with type 2 diabetes is below 7.0% per ADA 2024 standards.",
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

const AI_REPLY_POOL: MockReply[] = [
  {
    content:
      "The chest X-ray demonstrated mild cardiomegaly with a cardiothoracic ratio of 0.52. Both lung fields were clear with no consolidation, pleural effusion, or pneumothorax identified.",
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
      "The fasting blood glucose was 118 mg/dL, classifying as impaired fasting glucose per WHO criteria. The treating clinician recommended a repeat OGTT in 3 months alongside lifestyle intervention.",
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
      "Current medications include Metformin 500 mg PO twice daily and Atorvastatin 20 mg PO once nightly. A follow-up was scheduled 6 weeks post-assessment.",
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
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function formatTime(date: Date): string {
  const h = date.getHours().toString().padStart(2, "0");
  const m = date.getMinutes().toString().padStart(2, "0");
  return `${h}:${m}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// SVG Icon library (inline, zero deps)
// ─────────────────────────────────────────────────────────────────────────────

function IconEcg({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 80 28" className={className} fill="none" stroke="currentColor"
      strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      style={{ animation: "ecgDraw 2.4s ease-in-out infinite" }} aria-hidden="true">
      <polyline points="0,14 12,14 17,4 22,24 27,14 36,14 41,2 46,26 51,14 60,14 65,9 70,19 75,14 80,14" />
    </svg>
  );
}

function IconDocument({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function IconUpload({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 48 48" className={className} fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="6" y="30" width="36" height="12" rx="4" strokeOpacity="0.4" />
      <path d="M24 28V10" />
      <path d="M15 19l9-9 9 9" />
      <circle cx="38" cy="12" r="6" fill="currentColor" fillOpacity="0.12" strokeOpacity="0" />
      <path d="M35 12h6M38 9v6" strokeOpacity="0.5" />
    </svg>
  );
}

function IconSend({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M22 2L11 13" />
      <path d="M22 2L15 22l-4-9-9-4 20-7z" />
    </svg>
  );
}

function IconChevron({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 20 20" className={className} fill="currentColor" aria-hidden="true">
      <path fillRule="evenodd" clipRule="evenodd"
        d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" />
    </svg>
  );
}

function IconShield({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}

function IconWarning({ className }: { className?: string }): React.JSX.Element {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Premium Reference Card (replaces <details>)
// ─────────────────────────────────────────────────────────────────────────────

function ReferenceCard({ citation }: { citation: Citation }): React.JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="rounded-xl border transition-all duration-300 overflow-hidden"
      style={{
        background: open ? "rgba(225,245,254,0.55)" : "rgba(255,255,255,0.65)",
        borderColor: open ? "#0288d1" : "#e1f5fe",
        boxShadow: open ? "0 2px 12px 0 rgba(2,136,209,0.10)" : "none",
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-3 px-3.5 py-2.5 text-left group"
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <span
            className="shrink-0 w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold"
            style={{ background: "#e1f5fe", color: "#0288d1" }}
          >
            {citation.page}
          </span>
          <span className="font-mono text-[11px] font-semibold text-oceanic-blue truncate">
            {citation.chunkId}
          </span>
          <span className="text-[10px] text-calming-neutral-text hidden sm:block truncate" style={{ opacity: 0.45 }}>
            {citation.source}
          </span>
        </div>
        <IconChevron
          className={`w-3.5 h-3.5 shrink-0 text-oceanic-blue transition-transform duration-300 ${open ? "rotate-180" : ""}`}
        />
      </button>
      <div
        className="overflow-hidden transition-all duration-300"
        style={{ maxHeight: open ? "200px" : "0px", opacity: open ? 1 : 0 }}
      >
        <div className="px-3.5 pb-3 pt-0.5 border-t border-oceanic-blue-light">
          <blockquote
            className="text-xs leading-relaxed italic pl-3 border-l-2 border-oceanic-blue"
            style={{ color: "#212529", opacity: 0.72 }}
          >
            {citation.snippet}
          </blockquote>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Message bubble
// ─────────────────────────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: Message }): React.JSX.Element {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[76%] flex flex-col items-end gap-1">
          <div
            className="rounded-3xl rounded-tr-lg px-5 py-3.5 shadow-sm"
            style={{
              background: "linear-gradient(135deg, #e8f5e9 0%, #e1f5fe 100%)",
              color: "#1b5e20",
            }}
          >
            <p className="text-sm leading-relaxed font-medium">{message.content}</p>
          </div>
          <span className="text-[10px] pr-1" style={{ color: "#212529", opacity: 0.35 }}>
            You · {formatTime(message.timestamp)}
          </span>
        </div>
      </div>
    );
  }

  // AI message
  return (
    <div className="flex justify-start gap-3">
      {/* Avatar */}
      <div
        className="shrink-0 w-8 h-8 rounded-2xl flex items-center justify-center shadow-sm mt-0.5"
        style={{
          background: "linear-gradient(135deg, #2e7d32 0%, #0288d1 100%)",
        }}
      >
        <IconShield className="w-4 h-4 text-white" />
      </div>

      <div className="flex-1 max-w-[80%] flex flex-col gap-2">
        {/* Label */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-calming-neutral-text" style={{ opacity: 0.7 }}>
            Medical Assistant
          </span>
          <span className="text-[10px] text-calming-neutral-text" style={{ opacity: 0.35 }}>
            {formatTime(message.timestamp)}
          </span>
          {message.lowConfidence && (
            <span
              className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold"
              style={{ background: "#ffebee", color: "#c62828" }}
            >
              <IconWarning className="w-2.5 h-2.5" />
              Low confidence
            </span>
          )}
        </div>

        {/* Bubble */}
        <div
          className="rounded-3xl rounded-tl-lg px-5 py-4 shadow-sm"
          style={{
            background: message.lowConfidence
              ? "rgba(255,235,238,0.85)"
              : "rgba(255,255,255,0.88)",
            backdropFilter: "blur(8px)",
            border: message.lowConfidence
              ? "1px solid rgba(211,47,47,0.20)"
              : "1px solid rgba(233,236,239,0.8)",
          }}
        >
          <p className="text-sm leading-relaxed text-calming-neutral-text">
            {message.content}
          </p>

          {/* Reference cards */}
          {message.citations && message.citations.length > 0 && (
            <div className="mt-4 space-y-2">
              <p
                className="text-[10px] font-bold uppercase tracking-widest"
                style={{ color: "#0288d1", opacity: 0.8 }}
              >
                Source References · {message.citations.length} found
              </p>
              {message.citations.map((c) => (
                <ReferenceCard key={c.chunkId} citation={c} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Typing indicator
// ─────────────────────────────────────────────────────────────────────────────

function TypingIndicator(): React.JSX.Element {
  return (
    <div className="flex justify-start gap-3">
      <div
        className="shrink-0 w-8 h-8 rounded-2xl flex items-center justify-center shadow-sm"
        style={{ background: "linear-gradient(135deg, #2e7d32 0%, #0288d1 100%)" }}
      >
        <IconShield className="w-4 h-4 text-white" />
      </div>
      <div
        className="rounded-3xl rounded-tl-lg px-5 py-4 shadow-sm"
        style={{
          background: "rgba(255,255,255,0.88)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(233,236,239,0.8)",
        }}
      >
        <div className="flex gap-1.5 items-center h-4">
          {[0, 160, 320].map((delay) => (
            <span
              key={delay}
              className="w-2 h-2 rounded-full animate-bounce"
              style={{ background: "#2e7d32", animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Report item card
// ─────────────────────────────────────────────────────────────────────────────

function ReportItem({
  report,
  isActive,
  onClick,
}: {
  report: Report;
  isActive: boolean;
  onClick: () => void;
}): React.JSX.Element {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      id={`report-${report.id}`}
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="w-full text-left rounded-2xl px-4 py-3.5 transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-oceanic-blue"
      style={{
        background: isActive
          ? "linear-gradient(135deg, rgba(46,125,50,0.12) 0%, rgba(2,136,209,0.12) 100%)"
          : hovered
          ? "rgba(255,255,255,0.8)"
          : "rgba(255,255,255,0.5)",
        border: isActive
          ? "1.5px solid rgba(2,136,209,0.35)"
          : "1.5px solid transparent",
        boxShadow: isActive
          ? "0 2px 12px 0 rgba(2,136,209,0.12)"
          : hovered
          ? "0 2px 8px 0 rgba(0,0,0,0.06)"
          : "none",
        transform: hovered && !isActive ? "translateX(2px)" : "none",
      }}
    >
      <div className="flex items-center gap-3">
        {/* Icon badge */}
        <div
          className="shrink-0 w-9 h-9 rounded-xl flex items-center justify-center"
          style={{
            background: isActive
              ? "linear-gradient(135deg, #e8f5e9 0%, #e1f5fe 100%)"
              : "rgba(241,245,249,0.9)",
          }}
        >
          <IconDocument
            className="w-5 h-5"
            style={{ color: isActive ? "#0288d1" : "#64748b" } as React.CSSProperties}
          />
        </div>

        {/* Meta */}
        <div className="min-w-0 flex-1">
          <p
            className="text-xs font-semibold truncate"
            style={{ color: isActive ? "#01579b" : "#212529" }}
          >
            {report.name}
          </p>
          <p className="text-[10px] mt-0.5 flex items-center gap-1.5" style={{ color: "#212529", opacity: 0.45 }}>
            <span>{report.pages} pg</span>
            <span>·</span>
            <span>{report.chunks} chunks</span>
            <span>·</span>
            <span>{report.size}</span>
          </p>
        </div>

        {/* Status dot */}
        <span
          className="shrink-0 w-2 h-2 rounded-full"
          style={{ background: report.status === "indexed" ? "#2e7d32" : "#fbc02d" }}
          title={report.status === "indexed" ? "Indexed" : "Processing"}
        />
      </div>
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Stat tile
// ─────────────────────────────────────────────────────────────────────────────

function StatTile({
  value,
  label,
  accent,
}: {
  value: string;
  label: string;
  accent: string;
}): React.JSX.Element {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      className="flex-1 rounded-2xl px-4 py-3 flex flex-col items-center gap-0.5 transition-all duration-200"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? "rgba(255,255,255,0.92)" : "rgba(255,255,255,0.70)",
        border: `1.5px solid ${accent}30`,
        boxShadow: hovered ? `0 4px 16px 0 ${accent}22` : "none",
        transform: hovered ? "translateY(-2px)" : "none",
      }}
    >
      <p className="text-xl font-extrabold" style={{ color: accent }}>{value}</p>
      <p className="text-[9px] font-bold uppercase tracking-widest" style={{ color: "#212529", opacity: 0.45 }}>
        {label}
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Low Confidence Alert
// ─────────────────────────────────────────────────────────────────────────────

function LowConfidenceAlert(): React.JSX.Element {
  return (
    <div
      className="mx-4 mb-3 flex items-start gap-3 rounded-2xl px-4 py-3.5"
      role="alert"
      aria-live="polite"
      style={{
        background: "rgba(255,235,238,0.9)",
        backdropFilter: "blur(8px)",
        border: "1.5px solid rgba(211,47,47,0.25)",
        animation: "fadeSlideIn 0.28s ease-out",
      }}
    >
      <IconWarning className="w-5 h-5 shrink-0 mt-0.5 text-vitality-red" />
      <div>
        <p className="text-sm font-bold text-vitality-red-dark">Low Confidence Response</p>
        <p className="text-xs text-vitality-red-dark mt-0.5 leading-relaxed" style={{ opacity: 0.75 }}>
          The assistant could not reliably locate an answer within the indexed documents.
          Verify this information with a qualified healthcare professional before acting on it.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Medical Disclaimer
// ─────────────────────────────────────────────────────────────────────────────

function MedicalDisclaimer(): React.JSX.Element {
  return (
    <div
      className="px-4 py-3 flex items-start gap-2.5"
      role="note"
      aria-label="Medical disclaimer"
      style={{
        background: "linear-gradient(90deg, rgba(255,253,231,0.9) 0%, rgba(255,255,255,0.6) 100%)",
        borderTop: "1px solid rgba(251,192,45,0.35)",
      }}
    >
      <span className="text-gentle-yellow shrink-0 mt-0.5">⚕️</span>
      <p className="text-[10.5px] leading-relaxed text-gentle-yellow-dark">
        <strong>Educational Use Only.</strong> This tool does not constitute professional
        medical advice, diagnosis, or treatment. Always consult a qualified healthcare
        professional before making any medical decisions.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page component
// ─────────────────────────────────────────────────────────────────────────────

export default function Home(): React.JSX.Element {
  const [reports] = useState<Report[]>(SEED_REPORTS);
  const [activeReportId, setActiveReportId] = useState<string>("r1");
  const [messages, setMessages] = useState<Message[]>(SEED_MESSAGES);
  const [inputValue, setInputValue] = useState<string>("");
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [showLowConfidenceAlert, setShowLowConfidenceAlert] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [replyIndex, setReplyIndex] = useState<number>(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

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

    await new Promise<void>((resolve) =>
      setTimeout(resolve, 1100 + Math.random() * 900)
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
    if (reply.lowConfidence) setShowLowConfidenceAlert(true);
  }, [inputValue, isTyping, replyIndex]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  // Total stats
  const totalChunks = reports.reduce((s, r) => s + r.chunks, 0);
  const totalPages = reports.reduce((s, r) => s + r.pages, 0);

  return (
    <>
      {/* ── Keyframe animations ──────────────────────────────────────────── */}
      <style>{`
        @keyframes ecgDraw {
          0%   { stroke-dasharray: 300; stroke-dashoffset: 300; opacity: 0.45; }
          40%  { opacity: 1; }
          80%  { stroke-dashoffset: 0; }
          100% { stroke-dashoffset: 0; opacity: 0.45; }
        }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(-10px) scale(0.98); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes msgPop {
          from { opacity: 0; transform: translateY(8px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes pulseRing {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.6; transform: scale(1.15); }
        }
        .msg-pop    { animation: msgPop 0.24s cubic-bezier(0.22,1,0.36,1); }
        .pulse-ring { animation: pulseRing 2s ease-in-out infinite; }
        details summary::-webkit-details-marker { display: none; }
        * { box-sizing: border-box; }
      `}</style>

      {/* ── Canvas ───────────────────────────────────────────────────────── */}
      <div
        className="flex flex-col min-h-screen"
        style={{
          background:
            "linear-gradient(145deg, #f0fdf4 0%, #ecfeff 35%, #e1f5fe 65%, #f0fdf4 100%)",
          fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
        }}
      >

        {/* ════════════════════════════════════════════════════════════════
            HEADER
        ════════════════════════════════════════════════════════════════ */}
        <header
          className="sticky top-0 z-30"
          role="banner"
          style={{
            background: "rgba(255,255,255,0.75)",
            backdropFilter: "blur(20px) saturate(1.4)",
            borderBottom: "1px solid rgba(233,236,239,0.7)",
            boxShadow: "0 1px 24px 0 rgba(46,125,50,0.07)",
          }}
        >
          <div className="max-w-screen-xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-4">

            {/* Brand */}
            <div className="flex items-center gap-4">
              <div
                className="w-9 h-9 rounded-2xl flex items-center justify-center shadow-sm shrink-0"
                style={{ background: "linear-gradient(135deg, #2e7d32, #0288d1)" }}
              >
                <IconShield className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1
                  className="text-sm font-extrabold leading-tight tracking-tight"
                  style={{ color: "#01579b" }}
                >
                  Medical Intelligence Hub
                </h1>
                <div className="flex items-center gap-2 mt-0.5">
                  <IconEcg className="w-14 h-4 text-vitality-red" />
                  <span
                    className="text-[10px] font-medium"
                    style={{ color: "#212529", opacity: 0.45 }}
                  >
                    RAG · llama-3.3-70b-versatile
                  </span>
                </div>
              </div>
            </div>

            {/* Status indicators */}
            <div className="flex items-center gap-2">
              <div
                className="hidden sm:flex items-center gap-1.5 rounded-full px-3.5 py-1.5"
                style={{
                  background: "rgba(232,245,233,0.85)",
                  border: "1px solid rgba(46,125,50,0.2)",
                }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full pulse-ring"
                  style={{ background: "#2e7d32" }}
                />
                <span className="text-[11px] font-semibold text-healing-green-dark">
                  RAG Online
                </span>
              </div>
              <div
                className="hidden md:flex items-center gap-1.5 rounded-full px-3.5 py-1.5"
                style={{
                  background: "rgba(225,245,254,0.85)",
                  border: "1px solid rgba(2,136,209,0.2)",
                }}
              >
                <span className="text-[11px] font-semibold text-oceanic-blue-dark">
                  Groq Inference
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* ════════════════════════════════════════════════════════════════
            MAIN BODY
        ════════════════════════════════════════════════════════════════ */}
        <div
          className="flex-1 max-w-screen-xl mx-auto w-full px-4 sm:px-8 py-6 flex gap-5"
          style={{ minHeight: 0, height: "calc(100vh - 4rem)" }}
        >

          {/* ── LEFT PANEL ──────────────────────────────────────────────── */}
          <aside
            className="hidden lg:flex flex-col w-[17.5rem] shrink-0 gap-4"
            aria-label="Document panel"
          >

            {/* Upload Zone Card */}
            <div
              id="pdf-dropzone"
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); }}
              className="rounded-3xl p-5 flex flex-col items-center gap-3 text-center cursor-pointer transition-all duration-250 shrink-0"
              style={{
                background: isDragging
                  ? "linear-gradient(135deg, rgba(232,245,233,0.95) 0%, rgba(225,245,254,0.95) 100%)"
                  : "rgba(255,255,255,0.75)",
                backdropFilter: "blur(12px)",
                border: isDragging
                  ? "2px solid rgba(2,136,209,0.55)"
                  : "2px dashed rgba(2,136,209,0.28)",
                boxShadow: isDragging
                  ? "0 8px 32px 0 rgba(2,136,209,0.18)"
                  : "0 2px 16px 0 rgba(0,0,0,0.05)",
                transform: isDragging ? "scale(1.02)" : "scale(1)",
              }}
            >
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center"
                style={{
                  background: "linear-gradient(135deg, #e8f5e9 0%, #e1f5fe 100%)",
                }}
              >
                <IconUpload
                  className="w-8 h-8"
                  style={{ color: isDragging ? "#0288d1" : "#2e7d32" } as React.CSSProperties}
                />
              </div>
              <div>
                <p className="text-sm font-bold text-calming-neutral-text">
                  Drop PDF reports here
                </p>
                <p className="text-xs mt-1" style={{ color: "#212529", opacity: 0.48 }}>
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
              </div>
              <p
                className="text-[10px] font-medium rounded-full px-3 py-1"
                style={{
                  background: "rgba(225,245,254,0.7)",
                  color: "#0288d1",
                }}
              >
                PDF · Max 50 MB
              </p>
            </div>

            {/* Report list card */}
            <div
              className="flex-1 rounded-3xl p-4 flex flex-col gap-3 overflow-hidden"
              style={{
                background: "rgba(255,255,255,0.72)",
                backdropFilter: "blur(12px)",
                border: "1px solid rgba(233,236,239,0.8)",
                boxShadow: "0 2px 20px 0 rgba(0,0,0,0.05)",
              }}
            >
              <div className="flex items-center justify-between px-1 shrink-0">
                <h2
                  className="text-[10px] font-extrabold uppercase tracking-widest"
                  style={{ color: "#212529", opacity: 0.5 }}
                >
                  Indexed Documents
                </h2>
                <span
                  className="text-[10px] font-bold rounded-full px-2 py-0.5"
                  style={{ background: "#e1f5fe", color: "#0288d1" }}
                >
                  {reports.length}
                </span>
              </div>

              <div className="flex flex-col gap-1.5 overflow-y-auto flex-1">
                {reports.map((r) => (
                  <ReportItem
                    key={r.id}
                    report={r}
                    isActive={activeReportId === r.id}
                    onClick={() => setActiveReportId(r.id)}
                  />
                ))}
              </div>
            </div>

            {/* Stats tiles row */}
            <div className="flex gap-2 shrink-0">
              <StatTile value={String(totalChunks)} label="Chunks" accent="#0288d1" />
              <StatTile value={String(totalPages)} label="Pages" accent="#2e7d32" />
              <StatTile value={String(reports.length)} label="Docs" accent="#f57f17" />
            </div>
          </aside>

          {/* ── CHAT PANEL ───────────────────────────────────────────────── */}
          <main
            id="chat-panel"
            className="flex-1 flex flex-col rounded-3xl overflow-hidden"
            aria-label="Chat interface"
            style={{
              background: "rgba(255,255,255,0.68)",
              backdropFilter: "blur(20px) saturate(1.3)",
              border: "1px solid rgba(233,236,239,0.75)",
              boxShadow:
                "0 4px 40px 0 rgba(2,136,209,0.08), 0 1px 4px 0 rgba(0,0,0,0.04)",
            }}
          >
            {/* Chat top bar */}
            <div
              className="flex items-center justify-between px-6 py-4 shrink-0"
              style={{
                background:
                  "linear-gradient(90deg, rgba(232,245,233,0.55) 0%, rgba(225,245,254,0.55) 100%)",
                borderBottom: "1px solid rgba(233,236,239,0.65)",
              }}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-md shrink-0"
                  style={{
                    background: "linear-gradient(135deg, #2e7d32 0%, #0288d1 100%)",
                  }}
                >
                  <IconShield className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm font-extrabold" style={{ color: "#01579b" }}>
                    Medical Assistant
                  </p>
                  <p
                    className="text-[10px]"
                    style={{ color: "#212529", opacity: 0.42 }}
                  >
                    Powered by llama-3.3-70b-versatile · Groq
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                <span
                  className="w-2 h-2 rounded-full pulse-ring"
                  style={{ background: "#2e7d32" }}
                />
                <span className="text-xs font-semibold text-healing-green-dark">Active</span>
              </div>
            </div>

            {/* Messages area */}
            <div
              id="messages-area"
              className="flex-1 overflow-y-auto px-6 py-5 space-y-5 min-h-0"
              aria-live="polite"
              aria-relevant="additions"
            >
              {messages.map((msg) => (
                <div key={msg.id} className="msg-pop">
                  <MessageBubble message={msg} />
                </div>
              ))}
              {isTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>

            {/* Low confidence alert */}
            {showLowConfidenceAlert && <LowConfidenceAlert />}

            {/* Input zone */}
            <div className="shrink-0" style={{ borderTop: "1px solid rgba(233,236,239,0.65)" }}>
              <div className="px-5 pt-4 pb-2 flex gap-3 items-end">
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
                  className="flex-1 resize-none rounded-2xl px-4 py-3 text-sm transition-all duration-200 disabled:opacity-50 focus:outline-none"
                  style={{
                    background: "rgba(248,249,250,0.85)",
                    border: "1.5px solid rgba(233,236,239,0.9)",
                    color: "#212529",
                    boxShadow: "inset 0 1px 3px 0 rgba(0,0,0,0.04)",
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.border = "1.5px solid rgba(2,136,209,0.55)";
                    e.currentTarget.style.boxShadow = "0 0 0 3px rgba(2,136,209,0.10)";
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.border = "1.5px solid rgba(233,236,239,0.9)";
                    e.currentTarget.style.boxShadow = "inset 0 1px 3px 0 rgba(0,0,0,0.04)";
                  }}
                />

                {/* Send button */}
                <button
                  id="send-button"
                  type="button"
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isTyping}
                  aria-label="Send message"
                  className="shrink-0 w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-oceanic-blue focus-visible:ring-offset-2 disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: "linear-gradient(135deg, #2e7d32 0%, #0288d1 100%)",
                    boxShadow: "0 4px 14px 0 rgba(2,136,209,0.30)",
                  }}
                  onMouseEnter={(e) => {
                    if (!e.currentTarget.disabled) {
                      e.currentTarget.style.boxShadow = "0 6px 20px 0 rgba(2,136,209,0.40)";
                      e.currentTarget.style.transform = "translateY(-1px)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = "0 4px 14px 0 rgba(2,136,209,0.30)";
                    e.currentTarget.style.transform = "none";
                  }}
                >
                  <IconSend className="w-5 h-5 text-white" />
                </button>
              </div>

              {/* Keyboard hint */}
              <div className="px-5 pb-2 flex items-center gap-1.5">
                <kbd
                  className="text-[10px] rounded px-1.5 py-0.5 font-mono"
                  style={{
                    background: "rgba(233,236,239,0.7)",
                    color: "#212529",
                    opacity: 0.5,
                    border: "1px solid rgba(233,236,239,0.9)",
                  }}
                >
                  Enter
                </kbd>
                <span className="text-[10px]" style={{ color: "#212529", opacity: 0.35 }}>
                  send ·
                </span>
                <kbd
                  className="text-[10px] rounded px-1.5 py-0.5 font-mono"
                  style={{
                    background: "rgba(233,236,239,0.7)",
                    color: "#212529",
                    opacity: 0.5,
                    border: "1px solid rgba(233,236,239,0.9)",
                  }}
                >
                  Shift+Enter
                </kbd>
                <span className="text-[10px]" style={{ color: "#212529", opacity: 0.35 }}>
                  new line
                </span>
              </div>

              <MedicalDisclaimer />
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
