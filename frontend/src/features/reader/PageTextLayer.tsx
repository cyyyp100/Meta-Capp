import { useMemo } from "react";

import type { PageWord } from "../../api/types";
import { TEXT_LAYER_FONT, fitScaleX, measureText, placeWords } from "./textLayer";

/**
 * Calque de texte transparent posé sur l'image d'une page : c'est lui qui rend
 * le PDF sélectionnable et copiable, et qui porte les `data-wi` d'où le lecteur
 * tire les rects d'un surlignage.
 *
 * Chaque span est placé sur la boîte calculée par `placeWords` (bande de ligne
 * pour le texte horizontal, boîte étroite pour un tampon pivoté), puis comprimé
 * horizontalement pour la couvrir exactement : sans cette compression, la
 * largeur naturelle du mot dans la police du navigateur n'a aucune raison de
 * coïncider avec celle du PDF et la sélection dérive le long de la ligne.
 */
export function PageTextLayer({
  words,
  scale,
  tourAnchor = false,
}: {
  words: PageWord[];
  scale: number;
  /** Ancre de la visite guidée : c'est ce calque, et lui seul, qui rend la page
   *  sélectionnable — donc surlignable. L'étape qui explique le geste montre
   *  bien l'objet qui le permet, plutôt qu'une barre d'outils qui n'existe
   *  qu'une fois la sélection faite. Posée sur une seule page (cf. Reader). */
  tourAnchor?: boolean;
}) {
  // Le facteur de compression ne dépend pas du zoom (rapport de deux largeurs) :
  // on ne mesure qu'à l'arrivée des mots, pas à chaque changement d'échelle.
  const spans = useMemo(
    () =>
      placeWords(words).map((p) => {
        // Mot pivoté : l'em se lit sur la largeur de la boîte, le texte court sur sa hauteur.
        const fontSize = p.rotated ? p.width : p.height;
        const target = p.rotated ? p.height : p.width;
        return { ...p, fontSize, scaleX: fitScaleX(p.text, fontSize, target, measureText) };
      }),
    [words],
  );

  return (
    <div
      data-textlayer
      {...(tourAnchor ? { "data-tour": "sel-hint" } : {})}
      style={{
        position: "absolute",
        inset: 0,
        cursor: "text",
        userSelect: "text",
        WebkitUserSelect: "text",
      }}
    >
      {spans.map((s) => (
        <span
          key={s.wi}
          data-wi={s.wi}
          style={{
            position: "absolute",
            left: s.left * scale,
            // Pivoté : la rotation part du coin BAS-gauche de la boîte, le texte
            // remonte donc dans la colonne. Un tampon pivoté dans l'autre sens
            // couvrirait la même boîte, seul l'ordre des glyphes changerait.
            top: (s.rotated ? s.top + s.height : s.top) * scale,
            fontFamily: TEXT_LAYER_FONT,
            fontSize: s.fontSize * scale,
            lineHeight: 1,
            color: "transparent",
            whiteSpace: "pre",
            transformOrigin: "0 0",
            transform: s.rotated ? `rotate(-90deg) scaleX(${s.scaleX})` : `scaleX(${s.scaleX})`,
            userSelect: "text",
            WebkitUserSelect: "text",
          }}
        >
          {s.text}
        </span>
      ))}
    </div>
  );
}
