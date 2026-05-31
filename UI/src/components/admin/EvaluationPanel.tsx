import { Activity } from 'lucide-react';
import { useState } from 'react';
import { runEvaluation } from '../../api/admin';
import Button from '../ui/Button';

export default function EvaluationPanel() {
  const [result, setResult] = useState<{ score: number; passed: number; failed: number }>();
  const [isRunning, setIsRunning] = useState(false);

  const run = async () => {
    setIsRunning(true);
    try {
      setResult(await runEvaluation());
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <section className="admin-panel">
      <div className="panel-heading">
        <h2>Évaluation</h2>
        <Button onClick={() => void run()} disabled={isRunning}>
          <Activity size={18} />
          Lancer
        </Button>
      </div>
      {result ? (
        <div className="metrics">
          <span>Score: {Math.round(result.score * 100)}%</span>
          <span>OK: {result.passed}</span>
          <span>KO: {result.failed}</span>
        </div>
      ) : (
        <p className="empty">Aucun résultat disponible.</p>
      )}
    </section>
  );
}
