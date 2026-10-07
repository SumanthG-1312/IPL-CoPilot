"use client";

import { FormEvent, useState } from "react";
import { ArrowUp, Plus, Search } from "lucide-react";
import { motion } from "motion/react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type AskResponse = {
  session_id: string;
  answer: string;
};

const suggestions = [
  "Top run scorers",
  "Top wicket-takers",
  "Most sixes",
];

function CricketLogo() {
  return (
    <div className="relative h-12 w-16">
      <div className="absolute left-5 top-0 h-10 w-2.5 rotate-[38deg] rounded-full bg-gradient-to-b from-white via-slate-100 to-slate-300 shadow-[0_0_10px_rgba(255,255,255,0.12)]" />

      <div className="absolute left-2 top-6 h-4 w-8 rotate-[38deg] rounded-[3px] bg-gradient-to-r from-[#fffdf6] to-[#d7d7d7]" />

      <div className="absolute bottom-1 right-2 h-5 w-5 rounded-full bg-gradient-to-br from-white to-slate-300 shadow-[0_0_14px_rgba(255,255,255,0.18)]">
        <div className="absolute left-[22%] top-[38%] h-px w-[58%] rotate-[35deg] bg-slate-500/70" />
        <div className="absolute left-[22%] top-[53%] h-px w-[58%] rotate-[35deg] bg-slate-500/70" />
      </div>

      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "linear",
        }}
        className="absolute inset-0 rounded-full border border-cyan-400/25 border-r-transparent"
      />
    </div>
  );
}

function StadiumBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#02070B]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_42%,rgba(19,45,55,0.18),transparent_42%),linear-gradient(to_bottom,#02070B_0%,#020910_52%,#03100E_100%)]" />

      <motion.div
        animate={{
          opacity: [0.35, 0.5, 0.35],
          scale: [1, 1.06, 1],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute left-[-30px] top-[16%] h-40 w-40 rounded-full bg-white/[0.08] blur-[70px]"
      />

      <motion.div
        animate={{
          opacity: [0.35, 0.5, 0.35],
          scale: [1, 1.06, 1],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1.2,
        }}
        className="absolute right-[-30px] top-[16%] h-40 w-40 rounded-full bg-white/[0.08] blur-[70px]"
      />

      <div className="absolute left-0 top-[18%] h-[28%] w-[28%] origin-top-left -rotate-[8deg] bg-gradient-to-br from-white/[0.055] via-white/[0.015] to-transparent blur-xl" />

      <div className="absolute right-0 top-[18%] h-[28%] w-[28%] origin-top-right rotate-[8deg] bg-gradient-to-bl from-white/[0.055] via-white/[0.015] to-transparent blur-xl" />

      <div className="absolute left-1/2 top-[39%] h-[45%] w-[128%] -translate-x-1/2 rounded-[50%] border border-white/[0.035] bg-[radial-gradient(ellipse_at_center,rgba(12,32,40,0.7),rgba(2,10,14,0.96)_62%,#02070B_76%)]" />

      <div className="absolute left-1/2 top-[43%] h-[36%] w-[118%] -translate-x-1/2 rounded-[50%] border border-cyan-300/[0.025]" />

      <div className="absolute bottom-[20%] left-1/2 h-[25%] w-[112%] -translate-x-1/2 rounded-[50%] opacity-30 [background-image:radial-gradient(circle,rgba(255,255,255,0.65)_0.7px,transparent_1px)] [background-size:10px_10px]" />

      <div className="absolute bottom-[-18%] left-1/2 h-[52%] w-[102%] -translate-x-1/2 rounded-[50%] bg-[radial-gradient(ellipse_at_center,#143329_0%,#0B2119_40%,#04100D_72%)]" />

      <motion.div
        animate={{
          x: ["-10%", "10%", "-10%"],
          opacity: [0.04, 0.09, 0.04],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute bottom-[7%] left-1/2 h-20 w-[70%] -translate-x-1/2 rounded-full bg-cyan-200/[0.05] blur-2xl"
      />

      <div className="absolute bottom-[7%] left-1/2 h-[15%] w-[10%] -translate-x-1/2 rounded-[5px] bg-gradient-to-b from-[#C2A47C]/25 to-[#806747]/10" />

      <div className="absolute bottom-[10%] left-1/2 h-[10%] w-px -translate-x-1/2 bg-white/[0.08]" />

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_28%,rgba(0,0,0,0.28)_68%,rgba(0,0,0,0.72)_100%)]" />
    </div>
  );
}

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (value: string) => {
    const question = value.trim();

    if (!question || loading) return;

    setInput("");

    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: question,
      },
    ]);

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Request failed with status ${response.status}`
        );
      }

      const data: AskResponse = await response.json();

      setSessionId(data.session_id);

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      console.error("IPL Copilot API error:", error);

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            "I couldn't connect to the IPL Copilot backend. Make sure FastAPI is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    sendMessage(input);
  };

  const handleSuggestion = (question: string) => {
    setInput(question);
  };

  const newChat = () => {
    setMessages([]);
    setSessionId(null);
    setInput("");
  };

  const hasMessages = messages.length > 0;

  return (
    <main className="relative min-h-screen overflow-hidden text-white">
      <StadiumBackground />

      <div className="relative flex min-h-screen flex-col">
        <header className="flex items-center justify-between px-6 py-6 md:px-10">
          <motion.div
            initial={{ opacity: 0, x: -15 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="flex items-center gap-2"
          >
            <CricketLogo />

            <div className="-ml-1">
              <h1 className="text-[20px] font-semibold tracking-[-0.04em] md:text-[23px]">
                IPL{" "}
                <span className="bg-gradient-to-r from-sky-300 to-cyan-400 bg-clip-text text-transparent">
                  Copilot
                </span>
              </h1>

              <p className="mt-0.5 text-[9px] uppercase tracking-[0.24em] text-white/35">
                AI-powered IPL assistant
              </p>
            </div>
          </motion.div>

          <motion.button
            initial={{ opacity: 0, x: 15 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            type="button"
            onClick={newChat}
            className="flex items-center gap-2 rounded-full border border-white/[0.14] bg-black/20 px-4 py-2.5 text-sm text-white/80 shadow-[0_8px_30px_rgba(0,0,0,0.2)] backdrop-blur-2xl transition-all duration-300 hover:border-cyan-300/25 hover:bg-white/[0.05]"
          >
            <Plus size={16} />
            New Chat
          </motion.button>
        </header>

        <section className="flex flex-1 items-center justify-center px-5 pb-28 pt-4">
          <div className="w-full max-w-4xl">
            {!hasMessages ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.7,
                  ease: "easeOut",
                }}
              >
                <motion.div
                  initial="hidden"
                  animate="visible"
                  variants={{
                    hidden: {},
                    visible: {
                      transition: {
                        staggerChildren: 0.1,
                        delayChildren: 0.15,
                      },
                    },
                  }}
                  className="flex flex-wrap justify-center gap-3"
                >
                  {suggestions.map((suggestion) => (
                    <motion.button
                      key={suggestion}
                      type="button"
                      variants={{
                        hidden: {
                          opacity: 0,
                          y: 10,
                        },
                        visible: {
                          opacity: 1,
                          y: 0,
                        },
                      }}
                      whileHover={{ y: -2 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSuggestion(suggestion)}
                      className="rounded-full border border-white/[0.12] bg-[#071018]/55 px-7 py-3 text-sm font-medium text-white/80 shadow-[0_10px_35px_rgba(0,0,0,0.22)] backdrop-blur-xl transition-all duration-300 hover:border-cyan-300/25 hover:bg-[#0A1820]/65 hover:text-white"
                    >
                      {suggestion}
                    </motion.button>
                  ))}
                </motion.div>
              </motion.div>
            ) : (
              <div className="mx-auto flex max-h-[60vh] min-h-[420px] w-full max-w-3xl flex-col gap-4 overflow-y-auto px-1 pb-4">
                {messages.map((message, index) => (
                  <motion.div
                    key={`${message.role}-${index}`}
                    initial={{
                      opacity: 0,
                      y: 12,
                      scale: 0.98,
                    }}
                    animate={{
                      opacity: 1,
                      y: 0,
                      scale: 1,
                    }}
                    transition={{
                      duration: 0.3,
                    }}
                    className={
                      message.role === "user"
                        ? "ml-auto max-w-[85%] rounded-2xl rounded-br-md border border-cyan-300/20 bg-cyan-300/[0.06] px-5 py-4 text-sm text-white/90 shadow-[0_15px_45px_rgba(0,0,0,0.28)] backdrop-blur-2xl"
                        : "max-w-[85%] rounded-2xl rounded-bl-md border border-white/[0.10] bg-black/25 px-5 py-4 text-sm leading-6 text-white/75 shadow-[0_15px_45px_rgba(0,0,0,0.3)] backdrop-blur-2xl"
                    }
                  >
                    {message.content}
                  </motion.div>
                ))}

                {loading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="w-fit rounded-2xl rounded-bl-md border border-white/[0.10] bg-black/25 px-5 py-4 backdrop-blur-2xl"
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300" />
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300 [animation-delay:150ms]" />
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300 [animation-delay:300ms]" />
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>
        </section>

        <motion.div
          initial={{
            opacity: 0,
            y: 15,
          }}
          animate={{
            opacity: 1,
            y: 0,
          }}
          transition={{
            duration: 0.65,
            delay: 0.2,
          }}
          className="fixed bottom-7 left-0 right-0 px-5 md:px-10"
        >
          <form
            onSubmit={handleSubmit}
            className="mx-auto w-full max-w-5xl rounded-full border border-slate-300/[0.18] bg-[#071018]/55 p-2 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur-2xl transition-all duration-300 focus-within:border-cyan-300/30 focus-within:bg-[#071018]/65"
          >
            <div className="flex items-center gap-2">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center text-white/60">
                <Search size={25} strokeWidth={2} />
              </div>

              <div className="h-8 w-px bg-cyan-300/75 shadow-[0_0_10px_rgba(34,211,238,0.4)]" />

              <input
                type="text"
                value={input}
                disabled={loading}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    sendMessage(input);
                  }
                }}
                placeholder="Ask anything about the IPL..."
                className="h-12 flex-1 bg-transparent px-3 text-base text-white outline-none placeholder:text-white/38 disabled:cursor-not-allowed disabled:opacity-60"
              />

              <motion.button
                type="submit"
                disabled={!input.trim() || loading}
                whileHover={
                  input.trim() && !loading ? { scale: 1.04 } : {}
                }
                whileTap={
                  input.trim() && !loading ? { scale: 0.96 } : {}
                }
                className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-sky-300 to-blue-500 text-white shadow-[0_0_30px_rgba(59,130,246,0.18)] transition-all duration-300 hover:shadow-[0_0_42px_rgba(59,130,246,0.28)] disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ArrowUp size={20} strokeWidth={2.4} />
              </motion.button>
            </div>
          </form>

          <p className="mt-3 text-center text-[10px] text-white/25">
            IPL Copilot can make mistakes. Verify important statistics.
          </p>
        </motion.div>
      </div>
    </main>
  );
}