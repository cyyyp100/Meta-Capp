// TourHost.tsx — Le conducteur de la visite : navigation, démarrage, bulle.
//
// Monté UNE fois, au-dessus des routes (`App.tsx`) et non dans `AppLayout` :
// le lecteur et la séance de langue sont plein écran, donc hors du layout. Tant
// que l'hôte y vivait, les étapes du chapitre lecture devenaient bien actives
// mais rien ne les peignait jamais — trois bulles sur cinq ne s'étaient donc
// jamais affichées à personne.
//
// C'est aussi lui qui NAVIGUE. Le store est synchrone et sans routeur (il est
// appelé depuis un écouteur d'événement natif, hors de tout composant) ; le
// `useNavigate` de react-router n'existe que dans l'arbre React. La table de
// marche dit où chaque bulle se joue, cet hôte s'y rend.
import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { usePreferences } from "../shell/usePreferences";
import { Coachmark } from "./Coachmark";
import { TOUR_STEPS, resolveRoute, useTour } from "./useTour";

export function TourHost() {
  const navigate = useNavigate();
  const location = useLocation();
  const { data: preferences } = usePreferences();

  const running = useTour((s) => s.running);
  const index = useTour((s) => s.index);
  const demoDocId = useTour((s) => s.demoDocId);
  const hydrated = useTour((s) => s.hydrated);
  const done = useTour((s) => s.done);
  const hydrate = useTour((s) => s.hydrate);
  const start = useTour((s) => s.start);

  // L'état serveur fait autorité : le `localStorage` du webview ne survit pas à
  // une restauration de sauvegarde, et la visite recommencerait chez quelqu'un
  // qui l'a déjà faite.
  const storedDone = preferences?.preferences.tour_done;
  useEffect(() => {
    if (storedDone === undefined) return;
    hydrate(storedDone === "true");
  }, [storedDone, hydrate]);

  // Premier lancement : la visite part d'elle-même. C'est la seule bascule
  // automatique — partout ailleurs elle se demande.
  useEffect(() => {
    if (!hydrated || done || running) return;
    void start();
  }, [hydrated, done, running, start]);

  // Chaque étape sait où elle se joue ; on s'y rend si on n'y est pas déjà.
  const step = running ? TOUR_STEPS[index] : undefined;
  const route = step ? resolveRoute(step, demoDocId) : undefined;
  useEffect(() => {
    if (route && location.pathname !== route) navigate(route);
  }, [route, location.pathname, navigate]);

  if (!step) return null;
  return <Coachmark step={step} index={index} />;
}
