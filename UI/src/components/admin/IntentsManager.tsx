import { useEffect, useState } from 'react';
import { listIntents } from '../../api/admin';

interface IntentItem {
  id: string;
  name: string;
  examples: number;
  enabled: boolean;
}

export default function IntentsManager() {
  const [intents, setIntents] = useState<IntentItem[]>([]);

  useEffect(() => {
    void listIntents().then(setIntents).catch(() => setIntents([]));
  }, []);

  return (
    <section className="admin-panel">
      <h2>Intentions</h2>
      <div className="table-list">
        {intents.map((intent) => (
          <div className="table-row" key={intent.id}>
            <span>
              <strong>{intent.name}</strong>
              <small>{intent.examples} exemples</small>
            </span>
            <span className="status-pill">{intent.enabled ? 'Actif' : 'Inactif'}</span>
          </div>
        ))}
        {intents.length === 0 && <p className="empty">Aucune intention chargée.</p>}
      </div>
    </section>
  );
}
