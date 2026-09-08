// GemmaPanel.test.tsx — Les commandes du panneau doivent avoir un NOM.
//
// Elles étaient six emoji nus (✦ 🟢 ⚪️ 🎯 ⤢ ▭ ✕ ↵) : rien ne les annonçait à un
// lecteur d'écran, et leur rendu changeait d'un système à l'autre. Ce test fige
// le fait qu'on les atteint désormais par leur nom accessible.
//
// Le panneau ouvre un WebSocket vers /api/reader/{id}/stream : on le bouchonne,
// aucun serveur n'est nécessaire.

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { GemmaPanel } from "./GemmaPanel";

/** WebSocket minimal : se déclare ouvert, avale ce qu'on lui envoie. */
class FakeWebSocket {
  static readonly OPEN = 1;
  /** Dernier socket construit : sert à pousser un événement serveur au panneau. */
  static last: FakeWebSocket | null = null;
  readyState = FakeWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  sent: string[] = [];

  constructor(public url: string) {
    FakeWebSocket.last = this;
    // Ouverture asynchrone, comme un vrai socket.
    queueMicrotask(() => this.onopen?.());
  }

  /** Simule un message serveur (question, feedback…). */
  emit(event: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

/**
 * Rend le panneau ET l'ouvre. Gemma démarre replié en bulle ; l'ouvrir passait
 * par un `<div onClick>` — donc uniquement à la souris. C'est maintenant un
 * vrai bouton, ce que ce parcours vérifie au passage.
 */
async function renderOpenPanel() {
  const view = render(
    <TooltipProvider>
      <GemmaPanel docId={1} currentPage={1} sessionId={null} />
    </TooltipProvider>,
  );
  await userEvent.click(await screen.findByRole("button", { name: /gemma|ouvrir|open/i }));
  return view;
}

describe("GemmaPanel", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    // Le panneau mémorise sa position et son état ouvert/fermé dans localStorage.
    localStorage.clear();
  });

  it("s'ouvre au clavier et expose ses commandes par un nom accessible", async () => {
    await renderOpenPanel();

    // Ces trois commandes étaient 🎯, ⤢/▭ et ✕.
    expect(await screen.findByRole("button", { name: /focus/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /flottant|ancrer|float|dock/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /fermer|close/i })).toBeInTheDocument();

    // Le bouton d'envoi était un « ↵ » nu.
    expect(screen.getByRole("button", { name: /envoyer|send/i })).toBeInTheDocument();
  });

  it("annonce l'état de la connexion en toutes lettres", async () => {
    await renderOpenPanel();

    // C'était 🟢 / ⚪️ — deux emoji dont personne ne devine le sens.
    const light = await screen.findByRole("status");
    expect(light).toHaveAttribute("aria-label", expect.stringMatching(/gemma/i));
  });

  it("propose le mode d'accompagnement dans un sélecteur nommé et traduit", async () => {
    await renderOpenPanel();

    // Le <select> natif listait « discret / normal / coach » bruts.
    const trigger = await screen.findByRole("combobox", { name: /mode/i });
    expect(trigger).toBeInTheDocument();
    expect(trigger).toHaveTextContent(/normal/i);
    // Le bouton fermé annonce le mode, pas son explication : celle-ci n'aide
    // qu'au moment de choisir, et occupait un tiers de l'en-tête de Gemma.
    expect(trigger).not.toHaveTextContent(/moments clés|key moments/i);
  });

  it("neutralise les raccourcis pendant une réponse en cours", async () => {
    await renderOpenPanel();
    // Au repos, les raccourcis sont actionnables.
    const rephrase = await screen.findByRole("button", { name: /reformul|rephrase/i });
    expect(rephrase).toBeEnabled();
  });

  // Le type de question voyageait jusqu'ici sans rien changer à l'affichage :
  // une remise en ordre s'affichait comme une question ouverte, avec un champ
  // texte. La carte Q&R monte désormais le widget du type.
  it("monte le widget de remise en ordre quand la question l'exige", async () => {
    await renderOpenPanel();
    await act(async () => {
      FakeWebSocket.last?.emit({
        type: "qa_question",
        question: "Remets les étapes dans l'ordre.",
        question_type: "ordering",
        choices: ["Poser les hypothèses", "Appliquer le théorème", "Conclure"],
        mask: null,
      });
    });

    expect(await screen.findByText(/remise en ordre/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /valider/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /monter.*conclure/i })).toBeInTheDocument();
  });

  // `suggest_pause` traversait tout le serveur pour finir en phrase ordinaire
  // dans le fil : rien ne la distinguait, et rien ne permettait de la prendre.
  it("propose une pause qu'on peut réellement prendre", async () => {
    await renderOpenPanel();
    await act(async () => {
      FakeWebSocket.last?.emit({
        type: "intervention",
        kind: "suggest_pause",
        message: "Tes réponses raccourcissent : souffle deux minutes.",
        question: "",
        pause_minutes: 5,
        highlights: [],
      });
    });

    expect(await screen.findByText(/pause recommandée/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /faire une pause de 5 min/i }));

    // Le serveur doit l'apprendre : c'est lui qui suspend sa dérive d'attention.
    const sent = FakeWebSocket.last?.sent.map((raw) => JSON.parse(raw)) ?? [];
    expect(sent).toContainEqual({ type: "pause", minutes: 5 });
    // Et le décompte tourne, à la durée annoncée par le serveur.
    expect(screen.getByRole("timer")).toHaveTextContent("05:00");
  });

  // Le conseil de régulation de séance était produit par le modèle, validé par
  // le schéma… et jeté par le routeur. Il s'affiche avec la question.
  it("affiche le conseil de séance attaché à une question", async () => {
    await renderOpenPanel();
    await act(async () => {
      FakeWebSocket.last?.emit({
        type: "qa_question",
        question: "Que retiens-tu de ce passage ?",
        question_type: "open",
        choices: null,
        mask: null,
        session_hint: "Ton attention baisse : fais une pause courte avant de continuer.",
      });
    });

    expect(await screen.findByText(/fais une pause courte/i)).toBeInTheDocument();
  });

  it("signale le passage caché d'un rappel libre", async () => {
    await renderOpenPanel();
    await act(async () => {
      FakeWebSocket.last?.emit({
        type: "qa_question",
        question: "Redonne les trois éléments du passage.",
        question_type: "recall",
        choices: null,
        mask: { quote: "un passage à restituer", placeholder: "passage masqué" },
      });
    });

    expect(await screen.findByText(/rappel/i)).toBeInTheDocument();
    expect(screen.getByText(/masqué/i)).toBeInTheDocument();
  });
});
