import "./RiskBadge.css";

const LABELS = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical",
};

export default function RiskBadge({ level, pulse = false }) {
  return (
    <span className={`risk-badge risk-${level} ${pulse ? "risk-pulse" : ""}`}>
      <span className="risk-dot" />
      {LABELS[level] || level}
    </span>
  );
}
