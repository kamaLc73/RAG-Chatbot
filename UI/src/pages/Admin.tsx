import React, { useState } from 'react';
import {
  Activity,
  ChevronRight,
  FolderTree,
  MessageSquare,
  SearchCode,
  Shield,
  Target,
  Users,
} from 'lucide-react';
import ConversationsManager from '../components/admin/ConversationsManager';
import EvaluationPanel from '../components/admin/EvaluationPanel';
import IntentsManager from '../components/admin/IntentsManager';
import KBManager from '../components/admin/KBManager';
import UsersManager from '../components/admin/UsersManager';
import PageNavigation from '../components/layout/PageNavigation';
import type { User } from '../api/types';
import { cn } from '../utils/helpers';

type AdminTab = 'users' | 'conversations' | 'kb-management' | 'intentions' | 'retrieval-test' | 'evaluation';

interface TabErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class TabErrorBoundary extends React.Component<{ children: React.ReactNode }, TabErrorBoundaryState> {
  state: TabErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(error: Error): TabErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
          <p className="text-sm font-medium text-destructive">Une erreur est survenue dans ce panneau.</p>
          {this.state.error && (
            <pre className="max-w-lg overflow-auto rounded bg-muted p-3 text-left text-xs text-muted-foreground">
              {this.state.error.message}
            </pre>
          )}
          <button
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            onClick={() => this.setState({ hasError: false, error: undefined })}
          >
            Réessayer
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

const tabs = [
  {
    id: 'users' as AdminTab,
    label: 'Utilisateurs',
    icon: Users,
    description: 'Afficher et gérer les comptes utilisateur',
  },
  {
    id: 'conversations' as AdminTab,
    label: 'Conversations',
    icon: MessageSquare,
    description: 'Afficher les conversations et les messages',
  },
  {
    id: 'kb-management' as AdminTab,
    label: 'Base de connaissances',
    icon: FolderTree,
    description: 'Consulter les contenus Vespa et préparer les imports',
  },
  {
    id: 'intentions' as AdminTab,
    label: 'Intentions',
    icon: Target,
    description: 'Gérer les intents, les protections et les tests',
  },
  {
    id: 'retrieval-test' as AdminTab,
    label: 'Test récupération',
    icon: SearchCode,
    description: 'Tester la récupération hybride',
  },
  {
    id: 'evaluation' as AdminTab,
    label: 'Évaluation',
    icon: Activity,
    description: 'Suivre et lancer les évaluations RAGAS',
  },
];

function PlaceholderPanel({ title, description }: { title: string; description: string }) {
  return (
    <section className="admin-panel">
      <h2>{title}</h2>
      <p className="empty">{description}</p>
      <div className="mt-6 rounded-lg border border-dashed border-border bg-muted/40 p-8 text-center text-sm text-muted-foreground">
        Point d'accès en attente pour cette première version.
      </div>
    </section>
  );
}

export default function Admin() {
  const [activeTab, setActiveTab] = useState<AdminTab>('users');
  const [conversationsUser, setConversationsUser] = useState<User | null>(null);
  const currentTab = tabs.find((tab) => tab.id === activeTab) ?? tabs[0];

  const viewUserConversations = (user: User) => {
    setConversationsUser(user);
    setActiveTab('conversations');
  };

  const selectTab = (tab: AdminTab) => {
    if (tab === 'conversations') {
      setConversationsUser(null);
    }
    setActiveTab(tab);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'users':
        return <UsersManager onViewConversations={viewUserConversations} />;
      case 'conversations':
        return <ConversationsManager selectedUser={conversationsUser} onClearSelectedUser={() => setConversationsUser(null)} />;
      case 'kb-management':
        return (
          <div className="space-y-4">
            <KBManager />
            <PlaceholderPanel
              title="Import en masse"
              description="L'envoi et l'ingestion complète des documents, formulaires et vidéos seront gérés ici."
            />
          </div>
        );
      case 'intentions':
        return (
          <div className="space-y-4">
            <IntentsManager />
            <PlaceholderPanel
              title="Injections"
              description="Configuration des protections et des modèles de sécurité liés aux intentions."
            />
            <PlaceholderPanel
              title="Test des intents"
              description="Interface de test rapide à connecter au classificateur d'intentions."
            />
          </div>
        );
      case 'retrieval-test':
        return (
          <PlaceholderPanel
            title="Test récupération"
            description="Console de test de récupération Vespa à connecter aux points d'accès admin."
          />
        );
      case 'evaluation':
        return <EvaluationPanel />;
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="bg-card">
        <div className="mx-auto max-w-7xl border-b border-border px-4 py-4 sm:px-6">
          <div className="flex flex-col gap-4 sm:grid sm:grid-cols-[1fr_auto_1fr] sm:items-center">
            <div className="min-w-0">
              <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                <Shield className="h-4 w-4" />
                <span>Administrateur</span>
                <ChevronRight className="h-4 w-4" />
                <span className="font-medium text-foreground">{currentTab.label}</span>
              </div>
              <h1 className="text-2xl font-bold text-foreground">Panneau d'administration</h1>
              <p className="mt-1 text-sm text-muted-foreground">{currentTab.description}</p>
            </div>
            <div className="flex justify-center">
              <img src="/prevoyance_logo.png" alt="CDG Prévoyance" className="h-28 w-auto object-contain" />
            </div>
            <div className="flex sm:justify-end">
              <PageNavigation className="flex" />
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
          <div className="mb-8 flex flex-wrap gap-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;

              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => selectTab(tab.id)}
                  className={cn(
                    'inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                    isActive
                      ? 'bg-primary text-primary-foreground shadow-sm'
                      : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  )}
                  role="tab"
                  aria-selected={isActive}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div role="tabpanel" id={`${activeTab}-panel`}>
            <TabErrorBoundary key={activeTab}>{renderTabContent()}</TabErrorBoundary>
          </div>
        </div>
      </main>
    </div>
  );
}
