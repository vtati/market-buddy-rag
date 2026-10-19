"use client";

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";
import type { ChatResponse, ResolvedContext } from "@/lib/api";

export interface Turn {
  id: number;
  question: string;
  status: "pending" | "done" | "error";
  result?: ChatResponse;
  error?: string;
}

interface ChatSession {
  turns: Turn[];
  setTurns: React.Dispatch<React.SetStateAction<Turn[]>>;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  strategy: string;
  setStrategy: React.Dispatch<React.SetStateAction<string>>;
  mode: string;
  setMode: React.Dispatch<React.SetStateAction<string>>;
  corpusChoice: string;
  setCorpusChoice: React.Dispatch<React.SetStateAction<string>>;
  /** The last turn's resolved context — follow-ups bind to it. */
  context: ResolvedContext | null;
  setContext: React.Dispatch<React.SetStateAction<ResolvedContext | null>>;
}

const ChatSessionContext = createContext<ChatSession | null>(null);

/** Holds the chat conversation in the root layout so navigating to
 *  Add a Company / Lab / Eval and back does NOT wipe the transcript
 *  or the follow-up context. "New chat" is the explicit reset. The
 *  corpus is NOT part of this state — it lives server-side and is
 *  untouched by a new chat. */
export function ChatSessionProvider({ children }: { children: ReactNode }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [strategy, setStrategy] = useState("section");
  const [mode, setMode] = useState("hybrid+rerank");
  const [corpusChoice, setCorpusChoice] = useState("active");
  const [context, setContext] = useState<ResolvedContext | null>(null);

  return (
    <ChatSessionContext.Provider
      value={{
        turns, setTurns,
        input, setInput,
        strategy, setStrategy,
        mode, setMode,
        corpusChoice, setCorpusChoice,
        context, setContext,
      }}
    >
      {children}
    </ChatSessionContext.Provider>
  );
}

export function useChatSession(): ChatSession {
  const session = useContext(ChatSessionContext);
  if (!session) throw new Error("useChatSession outside ChatSessionProvider");
  return session;
}
