// api/queryClient.ts — Le client React Query de l'application, partagé.
//
// Il vivait dans `main.tsx`, ce qui le rendait inatteignable en dehors de
// l'arbre React. Or la visite guidée modifie la bibliothèque depuis un store
// Zustand — elle emprunte puis rend un document — et n'avait donc aucun moyen
// de prévenir le cache : la grille gardait la liste chargée au montage, et la
// carte de démonstration n'apparaissait jamais.
//
// Il est ici, et nulle part ailleurs : un second `QueryClient` donnerait deux
// caches, dont un que personne n'invalide.
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  // `refetchOnWindowFocus` désactivé : dans une fenêtre native, reprendre le
  // focus n'est pas le signal « je reviens après un long moment » qu'il est
  // dans un onglet de navigateur. Ce qui change, on l'invalide explicitement.
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});
