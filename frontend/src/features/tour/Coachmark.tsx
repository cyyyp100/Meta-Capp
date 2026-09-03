// Coachmark.tsx — Une bulle ancrée sur un vrai élément de l'interface.
//
// Rien de nouveau n'est installé pour ça : `popover.tsx` (Radix) sait déjà
// positionner une bulle sur un déclencheur en gérant le dépassement de
// viewport, et c'est exactement le primitif d'une coach mark. Une bibliothèque
// de tour apporterait son propre positionnement, son propre voile et son propre
// vocabulaire d'animation — trois doublons.
//
// Le voile est un `box-shadow` de très grand rayon posé sur un rectangle placé
// aux coordonnées de la cible : il assombrit tout SAUF la cible, sans jamais la
// recouvrir. Un vrai calque avec `clip-path` ferait le même effet mais
// intercepterait les clics ; ici la cible reste utilisable pendant que la bulle
// l'explique, ce qui est tout l'intérêt d'une coach mark.
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/ui/popover";

import { useT } from "../../i18n";
import type { TourStepDef } from "./steps";
import { TOUR_STEPS, useTour } from "./useTour";

/** Marge autour de la découpe : coller au pixel près donne un halo qui « pince »
 *  l'élément. */
const PADDING = 8;

/** Sondages infructueux (500 ms chacun) avant de renoncer à une cible.
 *
 *  Cinq secondes, et non deux : une étape qui change de route attend le
 *  chargement du morceau de code de l'écran PUIS la requête du document. Trop
 *  court, la visite sautait l'entrée du lecteur sur une machine lente — un saut
 *  qui ne dit pas son nom, donc le pire des deux comportements. */
const MISSING_LIMIT = 10;

export function Coachmark({ step, index }: { step: TourStepDef; index: number }) {
  const t = useT();
  const reduce = useReducedMotion();
  const next = useTour((s) => s.next);
  const skip = useTour((s) => s.skip);
  const [rect, setRect] = useState<DOMRect | null>(null);

  // La cible est résolue par `data-tour` : aucune `ref` à faire remonter, aucune
  // signature de composant modifiée pour la visite.
  //
  // Un attribut, une cible. `querySelector` prend le PREMIER du document : deux
  // éléments portant le même `data-tour` font pointer la bulle sur celui qui
  // est le plus haut dans le DOM, pas sur celui qu'on visait. La barre latérale
  // en portait, et les bulles désignaient le rail au lieu du bouton d'import et
  // du radar.
  useEffect(() => {
    // `measure()` juste en dessous repose ou efface le rectangle : pas besoin de
    // le remettre à zéro ici, ce qui déclencherait un rendu de plus par étape.
    let missing = 0;
    let scrolled = false;
    function measure() {
      const target = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
      if (target) {
        missing = 0;
        // La cible peut être hors du cadre : une bulle ancrée sur un élément
        // qu'on ne voit pas ne montre rien. UNE seule fois, à la découverte :
        // rejouer le défilement à chaque sondage empêcherait de bouger dans la
        // page pendant qu'on lit la bulle.
        if (!scrolled) {
          scrolled = true;
          target.scrollIntoView({ block: "nearest", inline: "nearest" });
        }
        setRect(target.getBoundingClientRect());
        return;
      }
      setRect(null);
      missing += 1;
      // Absente au bout de ~3 s : on PASSE à la suite plutôt que de rester
      // planté. Une ancre oubliée sur un écran doit coûter une bulle, jamais
      // la fin de la visite — c'est la seule panne qu'un parcours scripté
      // puisse rencontrer, et elle ne doit pas se voir.
      if (missing >= MISSING_LIMIT) next();
    }
    measure();
    // La cible bouge : défilement, redimensionnement, contenu qui se charge.
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    const timer = window.setInterval(measure, 500);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
      window.clearInterval(timer);
    };
  }, [step.target, next]);

  // Cible pas encore là : on n'affiche pas une bulle orpheline au milieu de
  // nulle part. Le sondage ci-dessus tranchera dans un sens ou dans l'autre.
  if (!rect || rect.width === 0) return null;

  const isLast = index === TOUR_STEPS.length - 1;

  return (
    <AnimatePresence>
      <motion.div
        key={step.id}
        className="pointer-events-none fixed inset-0 z-[120]"
        initial={reduce ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={reduce ? undefined : { opacity: 0 }}
        transition={{ duration: 0.28, ease: [0.33, 1, 0.68, 1] }}
      >
        {/* La découpe. `pointer-events: none` : la cible reste cliquable. */}
        <div
          aria-hidden
          className="absolute rounded-md"
          style={{
            top: rect.top - PADDING,
            left: rect.left - PADDING,
            width: rect.width + PADDING * 2,
            height: rect.height + PADDING * 2,
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.55)",
            transition: "top 0.2s, left 0.2s, width 0.2s, height 0.2s",
          }}
        />

        <Popover open>
          <PopoverAnchor asChild>
            <span
              aria-hidden
              className="absolute"
              style={{ top: rect.top, left: rect.left, width: rect.width, height: rect.height }}
            />
          </PopoverAnchor>
          <PopoverContent
            side={step.side ?? "right"}
            align="start"
            sideOffset={16}
            collisionPadding={16}
            // Radix rendrait le focus au déclencheur en se fermant : ici il n'y
            // a pas de déclencheur, et voler le focus retirerait le curseur du
            // champ dans lequel quelqu'un était peut-être en train d'écrire.
            onOpenAutoFocus={(event) => event.preventDefault()}
            onCloseAutoFocus={(event) => event.preventDefault()}
            // Radix PORTE la bulle dans `body` : elle sort du conteneur de la
            // coach mark et ne profite pas de son `z-120`. Avec le `z-50` par
            // défaut du primitif, elle passait donc SOUS le voile — dont
            // l'ombre de 9999 px la repeignait à 55 % de noir, texte compris —
            // et sous les sas (`z-100`), qui la masquaient entièrement pendant
            // les étapes du lecteur. Elle doit être au-dessus des deux.
            className="pointer-events-auto z-[130] w-80"
          >
            <p className="m-0 text-[11px] font-bold tracking-wide text-muted-foreground uppercase">
              {t(`tour.chapter.${step.chapter}`)} · {t("tour.step", { n: index + 1, total: TOUR_STEPS.length })}
            </p>
            <h3 className="mt-1.5 mb-0 font-serif text-h3 font-bold">{t(`tour.${step.id}.title`)}</h3>
            <p className="mt-2 mb-0 text-sm leading-relaxed text-text-soft">{t(`tour.${step.id}.body`)}</p>
            <div className="mt-4 flex items-center justify-between gap-3">
              {/* La visite est interruptible à TOUT moment, et le bouton pour en
                  sortir est aussi visible que celui pour continuer. */}
              <Button variant="ghost" size="sm" onClick={skip}>
                {t("tour.skip")}
              </Button>
              <Button size="sm" onClick={next}>
                {isLast ? t("tour.done") : t("tour.next")}
              </Button>
            </div>
          </PopoverContent>
        </Popover>
      </motion.div>
    </AnimatePresence>
  );
}
