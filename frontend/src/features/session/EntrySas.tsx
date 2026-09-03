import { useQuery } from "@tanstack/react-query";
import { Lightbulb } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";

import { api } from "../../api/client";
import { useT } from "../../i18n";
import { WhyButton } from "../science/WhyButton";
import { SasOverlay } from "./SasOverlay";
import { WarmUp } from "./WarmUp";

// SAS d'entrée : accroche de curiosité (LLM, 1 min, passable après 30 s) PUIS warm-up
// de 5 cartes sélectionnées par pertinence (clic-only), avant de démarrer la lecture.
//
// C'est un moment RITUEL, pas un écran d'attente : sa raison d'être est de faire
// ralentir avant de lire. D'où le cercle qui respire à la cadence d'une
// inspiration lente (≈5,5 s) et l'anneau qui se remplit — la durée devient
// perceptible au lieu d'être un simple chiffre qui décrémente.

const TOTAL_SECONDS = 60;
/** En dessous de ce reliquat, on peut passer à la suite. */
const SKIP_AT = 30;
/** Visite guidée : on montre le rituel, on ne l'impose pas. */
const DEMO_SECONDS = 8;

const RING_RADIUS = 46;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

export function EntrySas({
  docId,
  title,
  onStart,
  demo = false,
}: {
  docId: number;
  title: string;
  onStart: () => void;
  /**
   * Séance de démonstration de la visite guidée. Trois différences, toutes
   * pour la même raison — le sas est un rituel de RALENTISSEMENT, et on ne
   * ralentit pas quelqu'un qui découvre le produit :
   *   * le compte à rebours passe de 60 s à quelques secondes ;
   *   * l'accroche de curiosité est écrite d'avance (pas d'appel LLM, donc pas
   *     d'attente ni de dépendance à Ollama au premier lancement) ;
   *   * pas de warm-up : réviser des cartes n'a aucun sens sur un document
   *     qu'on n'a jamais lu, et l'utilisateur n'en a encore aucune.
   */
  demo?: boolean;
}) {
  const t = useT();
  const reduce = useReducedMotion();
  const [phase, setPhase] = useState<"intro" | "review">("intro");
  const totalSeconds = demo ? DEMO_SECONDS : TOTAL_SECONDS;
  // SAS de 1 minute, passable seulement après 30 s écoulées.
  const [left, setLeft] = useState(totalSeconds);
  const canSkip = demo || left <= SKIP_AT;

  const { data: hook } = useQuery({
    queryKey: ["hook", docId],
    queryFn: () => api.docHook(docId, 1),
    staleTime: Infinity,
    enabled: !demo,
  });
  // Warm-up : 5 cartes sélectionnées par pertinence (dues + récence × matière).
  const { data: cards } = useQuery({
    queryKey: ["session-start", docId],
    queryFn: () => api.sessionStartCards(docId),
    staleTime: Infinity,
    enabled: !demo,
  });
  const hookText = demo ? t("demo.entry_hook") : hook?.hook;

  // Compte à rebours (phase intro) : à 0, on passe au warm-up (pas direct à la lecture).
  useEffect(() => {
    if (phase !== "intro") return;
    if (left <= 0) {
      // En démonstration il n'y a pas de warm-up : on entre dans la lecture.
      if (demo) onStart();
      else setPhase("review");
      return;
    }
    const id = setTimeout(() => setLeft((l) => l - 1), 1000);
    return () => clearTimeout(id);
  }, [left, phase, demo, onStart]);

  // Warm-up sans carte disponible : on démarre la lecture directement.
  useEffect(() => {
    if (phase === "review" && cards && cards.length === 0) onStart();
  }, [phase, cards, onStart]);

  if (phase === "review") {
    if (!cards) {
      return (
        <SasOverlay contained>
          <div className="text-muted-foreground italic">{t("common.loading")}</div>
        </SasOverlay>
      );
    }
    if (cards.length > 0) return <WarmUp cards={cards} onDone={onStart} />;
    return <SasOverlay contained />;
  }

  const elapsed = totalSeconds - left;

  return (
    <SasOverlay contained>
      <motion.div
        className="max-w-[520px] px-8.5 text-center"
        initial={reduce ? false : { opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.33, 1, 0.68, 1] }}
      >
        <div className="mb-3 text-[13px] font-bold tracking-[1px] text-brand-ink uppercase">
          {t("entry.label")}
        </div>
        <h2 data-tour="entry" className="m-0 mb-2.5 font-serif text-2xl font-bold text-foreground">{title}</h2>
        <p className="leading-relaxed text-text-soft">{t("entry.text")}</p>
        <div className="mt-3">
          <WhyButton whyKey="entry" />
        </div>

        {hookText && (
          <motion.div
            className="mx-auto my-4 flex max-w-[460px] items-start gap-2.5 rounded-md bg-brand-soft px-4 py-3 text-left text-sm leading-relaxed text-accent-foreground"
            initial={reduce ? false : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.15, ease: [0.33, 1, 0.68, 1] }}
          >
            <Lightbulb className="mt-0.5 size-4 shrink-0" aria-hidden />
            <span>{hookText}</span>
          </motion.div>
        )}

        {/* Le compte à rebours n'était qu'un nombre dans un rond. L'anneau rend la
            durée visible d'un coup d'œil, et la respiration donne le tempo. */}
        <div className="relative mx-auto my-5.5 grid size-28 place-items-center">
          <motion.span
            aria-hidden
            className="absolute inset-0 rounded-full bg-brand-soft"
            animate={reduce ? undefined : { scale: [1, 1.08, 1], opacity: [0.5, 0.85, 0.5] }}
            transition={{ duration: 5.5, repeat: Infinity, ease: "easeInOut" }}
          />
          <svg className="absolute inset-0 size-28 -rotate-90" viewBox="0 0 100 100" aria-hidden>
            <circle
              cx="50"
              cy="50"
              r={RING_RADIUS}
              fill="none"
              stroke="var(--border)"
              strokeWidth="3"
            />
            <circle
              cx="50"
              cy="50"
              r={RING_RADIUS}
              fill="none"
              stroke="var(--accent)"
              strokeWidth="3"
              strokeLinecap="round"
              strokeDasharray={RING_CIRCUMFERENCE}
              strokeDashoffset={RING_CIRCUMFERENCE * (1 - elapsed / totalSeconds)}
              // Une seconde pile : l'anneau glisse au lieu de sauter par crans.
              style={{ transition: "stroke-dashoffset 1s linear" }}
            />
          </svg>
          <span
            role="timer"
            aria-live="off"
            className="relative text-3xl font-bold text-accent-foreground tabular-nums"
          >
            {left}
          </span>
        </div>

        <div className="flex flex-wrap justify-center gap-2.5">
          <Button size="lg" onClick={() => (demo ? onStart() : setPhase("review"))} disabled={!canSkip}>
            {canSkip ? t("entry.continue") : t("entry.skip_in", { n: left - SKIP_AT })}
          </Button>
        </div>
      </motion.div>
    </SasOverlay>
  );
}
