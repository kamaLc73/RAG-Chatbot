import { UploadCloud } from 'lucide-react';
import { useEffect, useState } from 'react';
import { listKbItems } from '../../api/admin';
import Button from '../ui/Button';

interface KbItem {
  id: string;
  title: string;
  source: string;
  status: string;
}

export default function KBManager() {
  const [items, setItems] = useState<KbItem[]>([]);

  useEffect(() => {
    void listKbItems().then(setItems).catch(() => setItems([]));
  }, []);

  return (
    <section className="admin-panel">
      <div className="panel-heading">
        <h2>Base documentaire</h2>
        <Button variant="secondary">
          <UploadCloud size={18} />
          Importer
        </Button>
      </div>
      <div className="table-list">
        {items.map((item) => (
          <div className="table-row" key={item.id}>
            <span>
              <strong>{item.title}</strong>
              <small>{item.source}</small>
            </span>
            <span className="status-pill">{item.status}</span>
          </div>
        ))}
        {items.length === 0 && <p className="empty">Aucun document chargé.</p>}
      </div>
    </section>
  );
}
