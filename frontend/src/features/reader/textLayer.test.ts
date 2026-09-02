import { describe, expect, it } from "vitest";

import type { PageWord } from "../../api/types";
import { fitScaleX, placeWords, placedBoxes } from "./textLayer";

// Boîtes réelles de la page 1 d'un arXiv (points PDF, origine haut-gauche) :
// le tampon vertical de la marge gauche, puis trois mots d'une ligne du résumé.
const STAMP: PageWord = [18.1, 396.3, 32.3, 559.3, "arXiv:2410.13228v1"];
const LINE: PageWord[] = [
  [100, 158.9, 141.2, 167.2, "Nazanin"],
  [144, 157.2, 219.4, 169.5, "Daryakenarib,c,"],
  [221, 165.9, 222.3, 169.4, ","],
];

describe("placeWords", () => {
  it("reconnaît un tampon pivoté et lui garde sa boîte étroite", () => {
    const [stamp] = placeWords([STAMP]);
    expect(stamp.rotated).toBe(true);
    expect(stamp.width).toBeCloseTo(14.2, 1);
    expect(stamp.height).toBeCloseTo(163, 1);
  });

  it("n'agrège pas le tampon aux lignes qu'il traverse", () => {
    const placed = placeWords([STAMP, ...LINE]);
    const band = placed.filter((p) => !p.rotated);
    expect(band).toHaveLength(3);
    // La bande reste celle du texte courant : ~12 pt, pas les 163 pt du tampon.
    expect(band.every((p) => p.height < 15)).toBe(true);
  });

  it("aligne les mots d'une ligne sur une bande commune", () => {
    const placed = placeWords(LINE);
    const tops = new Set(placed.map((p) => p.top.toFixed(3)));
    const heights = new Set(placed.map((p) => p.height.toFixed(3)));
    expect(tops.size).toBe(1);
    expect(heights.size).toBe(1);
    // Bande médiane des deux mots de taille courante — la virgule, hors
    // gabarit, l'adopte au lieu de former une ligne de 3 pt pour elle seule.
    expect(placed[0].height).toBeCloseTo(10.3, 1);
    expect(placed[0].top + placed[0].height).toBeCloseTo(168.35, 2);
  });

  it("garde à chaque mot sa position et sa largeur horizontales", () => {
    const placed = placeWords(LINE);
    expect(placed.map((p) => p.left)).toEqual([100, 144, 221]);
    expect(placed[1].width).toBeCloseTo(75.4, 1);
  });

  it("ignore un glyphe démesuré dans le calcul de la bande", () => {
    // Une grande parenthèse de formule ne doit pas étirer la bande sur les
    // lignes voisines : la médiane l'écarte.
    const withBracket: PageWord[] = [...LINE, [230, 150, 234, 178, "("]];
    const placed = placeWords(withBracket);
    expect(placed.every((p) => p.height < 15)).toBe(true);
  });

  it("ne prend pas une ligature étroite pour du texte pivoté", () => {
    const narrow: PageWord[] = [[10, 100, 15.5, 107.3, "fi"], [20, 100, 27, 110, "The"]];
    expect(placeWords(narrow).every((p) => !p.rotated)).toBe(true);
  });
});

describe("placedBoxes", () => {
  it("indexe les boîtes finales par index de mot d'origine", () => {
    const boxes = placedBoxes([STAMP, ...LINE]);
    expect(boxes.size).toBe(4);
    expect(boxes.get(0)?.[3]).toBeCloseTo(559.3, 1); // le tampon, boîte inchangée
    const [x0, y0, x1, y1] = boxes.get(3)!; // la virgule, remontée sur la bande
    expect([x0, x1]).toEqual([221, 222.3]);
    expect(y1 - y0).toBeCloseTo(10.3, 1);
  });
});

describe("fitScaleX", () => {
  it("comprime le span à la largeur de sa boîte", () => {
    expect(fitScaleX("Nazanin", 8.3, 41.2, () => 50)).toBeCloseTo(0.824, 3);
  });

  it("reste neutre quand la mesure est indisponible", () => {
    expect(fitScaleX("Nazanin", 8.3, 41.2, () => 0)).toBe(1);
  });
});
