import { Loader2, Mic, Send, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { transcribeAudio } from '../../api/chat';
import Button from '../ui/Button';

interface MessageInputProps {
  disabled?: boolean;
  placeholder?: string;
  onSend: (message: string) => void;
}

export default function MessageInput({ disabled, placeholder = 'Posez une question...', onSend }: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const resizeTextarea = () => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = '56px';
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 144)}px`;
  };

  const appendTranscription = (text: string) => {
    setMessage((current) => [current, text].filter(Boolean).join(' '));
    requestAnimationFrame(resizeTextarea);
  };

  const stopMediaStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  const getSupportedMimeType = () => {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
    return types.find((type) => MediaRecorder.isTypeSupported(type));
  };

  useEffect(() => {
    return () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.onstop = null;
        recorder.stop();
      }
      stopMediaStream();
    };
  }, []);

  const submit = () => {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setMessage('');
    if (textareaRef.current) {
      textareaRef.current.style.height = '56px';
    }
  };

  const transcribeRecording = async (mimeType?: string) => {
    const chunks = chunksRef.current;
    chunksRef.current = [];
    recorderRef.current = null;
    stopMediaStream();
    setIsRecording(false);

    if (chunks.length === 0) return;

    setIsTranscribing(true);
    try {
      const type = mimeType || chunks[0]?.type || 'audio/webm';
      const blob = new Blob(chunks, { type });
      const extension = type.includes('mp4') ? 'mp4' : 'webm';
      const file = new File([blob], `microphone-recording.${extension}`, { type });
      const result = await transcribeAudio(file);
      const transcript = result.text?.trim();
      if (!transcript) {
        setMessage((current) => current || 'Audio non transcrit. Veuillez saisir votre question.');
        return;
      }
      if (disabled) {
        appendTranscription(transcript);
        return;
      }
      onSend(transcript);
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '56px';
      }
    } catch {
      setMessage((current) => current || 'Audio non transcrit. Veuillez saisir votre question.');
    } finally {
      setIsTranscribing(false);
    }
  };

  const startRecording = async () => {
    if (disabled || isTranscribing) return;

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setMessage((current) => current || 'Microphone non supporté par ce navigateur.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        void transcribeRecording(recorder.mimeType);
      };
      recorder.onerror = () => {
        setIsRecording(false);
        stopMediaStream();
        setMessage((current) => current || 'Enregistrement audio interrompu.');
      };

      recorder.start();
      setIsRecording(true);
    } catch {
      stopMediaStream();
      setMessage((current) => current || 'Autorisation microphone refusée ou indisponible.');
    }
  };

  const stopRecording = () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;
    recorder.stop();
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
      return;
    }
    void startRecording();
  };

  return (
    <div className="p-4">
      <div className="rounded-lg border border-primary/20 bg-background p-3 shadow-sm">
        <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-stretch gap-3">
          <Button
            variant={isRecording ? 'destructive' : 'outline'}
            type="button"
            title={isRecording ? "Arrêter l'enregistrement" : 'Enregistrer avec le microphone'}
            aria-label={isRecording ? "Arrêter l'enregistrement" : 'Enregistrer avec le microphone'}
            onClick={toggleRecording}
            disabled={(!isRecording && disabled) || isTranscribing}
            className="h-full w-14 p-0"
          >
            {isTranscribing ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : isRecording ? (
              <Square className="h-4 w-4 fill-current" />
            ) : (
              <Mic className="h-5 w-5" />
            )}
          </Button>
          <textarea
            ref={textareaRef}
            value={message}
            placeholder={placeholder}
            rows={1}
            className="min-h-14 max-h-36 w-full resize-none rounded-md border border-input bg-background px-3 py-4 text-sm outline-none placeholder:text-muted-foreground focus:border-primary focus:ring-2 focus:ring-primary/20"
            onChange={(event) => {
              setMessage(event.target.value);
              requestAnimationFrame(resizeTextarea);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            disabled={disabled}
          />
          <Button type="button" onClick={submit} disabled={disabled || !message.trim()} aria-label="Envoyer" className="h-full w-14 p-0">
            <Send className="h-5 w-5" />
          </Button>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          {isRecording
            ? "Enregistrement en cours... cliquez sur le carré pour arrêter."
            : isTranscribing
              ? 'Transcription audio en cours...'
              : 'Appuyez sur Entrée pour envoyer, Maj + Entrée pour une nouvelle ligne'}
        </p>
      </div>
    </div>
  );
}