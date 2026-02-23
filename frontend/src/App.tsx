/**
 * Beat Studio — root application with 5-tab navigation.
 *
 * Tabs:
 *   1. Upload & Song Analysis
 *   2. Mashup Workshop
 *   3. Video Studio
 *   4. LoRA Manager
 *   5. System Status
 */
import { useState } from 'react';
import { AudioUpload } from './components/AudioUpload';
import { SongAnalysis } from './components/SongAnalysis';
import { MashupWorkshop } from './components/MashupWorkshop';
import { MashupLibrary } from './components/MashupLibrary';
import { VideoStudio } from './components/VideoStudio';
import { LoRAManager } from './components/LoRAManager';
import { NovaFadeStudio } from './components/NovaFadeStudio';
import { HardwareStatus } from './components/HardwareStatus';
import type { SectionInfo, SongAnalysis as SongAnalysisType } from './types';
import axios from 'axios';

type Tab = 'upload' | 'mashup' | 'video' | 'lora' | 'system';

interface AppState {
  audioId: string | null;
  analysis: SongAnalysisType | null;
  sections: SectionInfo[] | null;   // user-edited sections from SongAnalysis
  mashupId: string | null;
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'upload', label: 'Upload & Song' },
  { id: 'mashup', label: 'Mashup Workshop' },
  { id: 'video', label: 'Video Studio' },
  { id: 'lora', label: 'LoRA Manager' },
  { id: 'system', label: 'System Status' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('upload');
  const [state, setState] = useState<AppState>({
    audioId: null,
    analysis: null,
    sections: null,
    mashupId: null,
  });
  const [analyzing, setAnalyzing] = useState(false);

  const handleUploadComplete = async (audioId: string) => {
    setState(prev => ({ ...prev, audioId }));
    setAnalyzing(true);
    try {
      await axios.post('/api/audio/analyze', { audio_id: audioId });
      // Poll until analysis complete (background task)
      for (let i = 0; i < 60; i++) {
        await new Promise(r => setTimeout(r, 2000));
        const { data } = await axios.get<any>(`/api/audio/analysis/${audioId}`);
        if (data.status === 'complete' && data.analysis) {
          setState(prev => ({ ...prev, analysis: data.analysis }));
          break;
        }
      }
    } catch { /* non-fatal */ } finally {
      setAnalyzing(false);
    }
  };

  const handleMashupCreated = (taskId: string) => {
    setState(prev => ({ ...prev, mashupId: taskId }));
  };

  return (
    <div className="app" data-testid="app">
      <header className="app-header" data-testid="app-header">
        <h1>Beat Studio</h1>
        <p className="tagline">AI-powered music video production</p>
      </header>

      {/* Tab navigation */}
      <nav className="tab-nav" data-testid="tab-nav">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            data-testid={`tab-${tab.id}`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Tab content */}
      <main className="tab-content" data-testid="tab-content">

        {/* Tab 1: Upload & Song Analysis */}
        {activeTab === 'upload' && (
          <div data-testid="tab-upload">
            <AudioUpload
              onUploadComplete={(audioId) => void handleUploadComplete(audioId)}
            />
            {analyzing && <p data-testid="analyzing-msg">Analyzing audio…</p>}
            {state.analysis && (
              <SongAnalysis
                analysis={state.analysis}
                onCreateVideo={() => setActiveTab('video')}
                onCreateMashup={() => setActiveTab('mashup')}
                onSectionsChange={sections => setState(prev => ({ ...prev, sections }))}
              />
            )}
          </div>
        )}

        {/* Tab 2: Mashup Workshop */}
        {activeTab === 'mashup' && (
          <div data-testid="tab-mashup">
            <MashupLibrary onSongSelect={() => {}} />
            <MashupWorkshop
              initialSongId={state.audioId ?? undefined}
              onMashupCreated={handleMashupCreated}
              onMakeVideo={(id) => {
                setState(prev => ({ ...prev, audioId: id }));
                setActiveTab('video');
              }}
              onMakeDJVideo={(id) => {
                setState(prev => ({ ...prev, mashupId: id }));
                setActiveTab('lora');
              }}
            />
          </div>
        )}

        {/* Tab 3: Video Studio */}
        {activeTab === 'video' && (
          <div data-testid="tab-video">
            {state.audioId ? (
              <VideoStudio
                audioId={state.audioId}
                analysis={state.analysis ?? undefined}
                sections={state.sections ?? state.analysis?.sections ?? undefined}
                onBack={() => setActiveTab('upload')}
              />
            ) : (
              <div data-testid="no-audio-msg">
                <p>Upload a song first to create a video.</p>
                <button onClick={() => setActiveTab('upload')}>Go to Upload</button>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: LoRA Manager */}
        {activeTab === 'lora' && (
          <div data-testid="tab-lora">
            <LoRAManager />
            <NovaFadeStudio mashupId={state.mashupId ?? undefined} />
          </div>
        )}

        {/* Tab 5: System Status */}
        {activeTab === 'system' && (
          <div data-testid="tab-system">
            <HardwareStatus />
            <div className="system-info" data-testid="system-info">
              <h3>System Information</h3>
              <p>API: <a href="/api/system/health" target="_blank" rel="noreferrer">/api/system/health</a></p>
              <p>Docs: <a href="/docs" target="_blank" rel="noreferrer">/docs</a></p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
