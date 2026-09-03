// steps.ts — Le SCRIPT de la visite guidée.
//
// La visite était une poignée de bulles opportunistes : chaque écran signalait
// « mon contexte vient d'apparaître » et le store décidait s'il y avait une
// bulle à jouer. C'était élégant et ça ne montrait presque rien — quelqu'un qui
// n'importait aucun document ne voyait jamais ni Gemma, ni les sas, ni le
// profil, c'est-à-dire ni le produit ni ce qui le distingue.
//
// La visite est donc maintenant un PARCOURS : elle emprunte un document de
// démonstration (`services/onboarding.py`), traverse elle-même les écrans, et
// joue une fausse session de lecture de bout en bout. « Fausse » au sens fort :
// aucune requête de session, aucun WebSocket, aucun appel LLM — rien de ce qui
// se passe ici n'atteint le profil métacognitif (cf. `demoScript.ts`).
//
// Ce fichier ne contient QUE la table de marche. Les textes vivent dans i18n
// (`tour.<id>.title` / `tour.<id>.body`), la mécanique dans `useTour.ts`.
import type { TourContext } from "./useTour";

/** Les quatre temps du récit, affichés au-dessus du compteur d'étapes. */
export type TourChapter = "library" | "reading" | "profile" | "tools";

export interface TourStepDef {
  /** Sert de clé i18n ET d'identité de l'étape. */
  id: string;
  /** Où la bulle doit être jouée. La visite y navigue elle-même si besoin. */
  route?: string;
  /** Valeur de l'attribut `data-tour` de la cible. */
  target: string;
  /** Côté d'ancrage de la bulle. `right` par défaut. */
  side?: "top" | "right" | "bottom" | "left";
  chapter: TourChapter;
  /** Joué à l'ENTRÉE de l'étape : ouvrir un panneau, poser un message… */
  enter?: (ctx: TourContext) => void;
  /**
   * Étape qui n'a de sens que si le document de démonstration a pu être
   * emprunté. Sans lui, tout le chapitre lecture est sauté d'un bloc plutôt que
   * d'échouer bulle par bulle : une visite incomplète vaut mieux qu'une visite
   * qui s'arrête.
   */
  needsDemo?: boolean;
}

/**
 * La route du lecteur dépend de l'id prêté par le serveur, qu'on ne connaît
 * qu'au lancement de la visite. `:demo` est remplacé au moment de naviguer.
 */
export const DEMO_ROUTE = "/reader/:demo";

export const TOUR_STEPS: TourStepDef[] = [
  // ── Chapitre 1 : la bibliothèque ────────────────────────────────────────
  { id: "welcome", route: "/", target: "brand", side: "right", chapter: "library" },
  { id: "library", route: "/", target: "grid", side: "top", chapter: "library" },
  { id: "doc-card", route: "/", target: "doc-card", side: "right", chapter: "library", needsDemo: true },
  { id: "folders", route: "/", target: "folders", side: "right", chapter: "library" },
  { id: "search", route: "/", target: "search", side: "bottom", chapter: "library" },
  { id: "import", route: "/", target: "import", side: "bottom", chapter: "library" },

  // ── Chapitre 2 : une séance de lecture ──────────────────────────────────
  // `enter` du premier pas du chapitre : c'est la navigation vers le lecteur
  // qui monte le sas d'entrée, on n'a rien à jouer nous-mêmes.
  { id: "entry-sas", route: DEMO_ROUTE, target: "entry", side: "right", chapter: "reading", needsDemo: true },
  {
    id: "page",
    target: "page",
    side: "right",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.enterReading(),
  },
  { id: "toolbar", target: "toolbar", side: "bottom", chapter: "reading", needsDemo: true },
  { id: "select", target: "sel-hint", side: "bottom", chapter: "reading", needsDemo: true },
  { id: "gemma-bubble", target: "gemma", side: "left", chapter: "reading", needsDemo: true },
  {
    id: "gemma-panel",
    target: "gemma-body",
    side: "left",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.openPanel(),
  },
  {
    id: "gemma-answer",
    target: "gemma-body",
    side: "left",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.play("answer"),
  },
  { id: "gemma-mode", target: "gemma-mode", side: "bottom", chapter: "reading", needsDemo: true },
  { id: "gemma-chips", target: "gemma-chips", side: "top", chapter: "reading", needsDemo: true },
  {
    id: "intervention",
    target: "gemma-body",
    side: "left",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.play("intervention"),
  },
  {
    id: "gemma-qa",
    target: "gemma-qa",
    side: "left",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.play("question"),
  },
  {
    id: "exit-sas",
    target: "exit",
    side: "right",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.endSession(),
  },
  {
    id: "rest-sas",
    target: "rest",
    side: "right",
    chapter: "reading",
    needsDemo: true,
    enter: (ctx) => ctx.demo?.closeExitSas(),
  },

  // ── Chapitre 3 : ce que l'application a retenu ──────────────────────────
  // On rend le document de démonstration ICI, en quittant le lecteur : la
  // bibliothèque doit être vide quand on y revient à la dernière étape.
  { id: "profil", route: "/stats", target: "profil", side: "left", chapter: "profile" },
  { id: "progress", route: "/stats/progress", target: "progress", side: "bottom", chapter: "profile" },

  // ── Chapitre 4 : le reste de l'atelier ──────────────────────────────────
  { id: "flashcards", route: "/flashcards", target: "nav-flashcards", side: "right", chapter: "tools" },
  { id: "quiz", route: "/quiz", target: "nav-quiz", side: "right", chapter: "tools" },
  { id: "lang", route: "/lang", target: "nav-lang", side: "right", chapter: "tools" },
  { id: "brainstorming", route: "/brainstorming", target: "nav-brainstorming", side: "right", chapter: "tools" },
  { id: "settings", route: "/settings", target: "nav-settings", side: "right", chapter: "tools" },

  // ── Dernier mot : la bibliothèque, vide, et le geste à faire ────────────
  { id: "done", route: "/", target: "import", side: "bottom", chapter: "tools" },
];
