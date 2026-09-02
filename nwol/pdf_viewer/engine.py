# pdf_viewer/engine.py — Accès sérialisé au moteur PDFium
#
# **PDFium n'est pas thread-safe, et son état est GLOBAL au processus.** Deux
# threads qui ouvrent chacun leur document en même temps ne se gênent pas : ils
# renvoient un « Data format error » sur un fichier parfaitement valide, ou font
# carrément tomber le processus (`Abort trap: 6`). PyMuPDF tolérait cet usage,
# PDFium non — c'est le seul point où le remplacement n'était pas un swap 1-pour-1.
#
# Le backend est mono-thread par conception côté SQLite et côté file LLM, mais
# **pas côté HTTP** : FastAPI exécute chaque endpoint synchrone dans un thread du
# pool, et le lecteur réclame plusieurs pages en parallèle dès qu'on scrolle
# (`page/N.png`, `page/N/words`). D'où ce verrou unique.
#
# Il est tenu pendant **toute la durée d'un accès** (ouverture → fermeture), pas
# seulement autour d'un appel : les handles de page et de texte ne restent
# valides que tant qu'aucun autre thread n'entre dans la bibliothèque.
#
# Même politique que la file LLM sérialisée (`llm/ollama_client.py`) : un seul
# consommateur d'une ressource native à la fois. Le coût est négligeable — un
# rendu de page A4 tient en ~0,03 s.
from __future__ import annotations

import threading

# RLock et non Lock : un même thread peut légitimement imbriquer deux accès
# (un rendu de page demandé au milieu de l'indexation d'un document).
PDFIUM_LOCK = threading.RLock()
