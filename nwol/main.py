#!/usr/bin/env python3
# main.py — Point d'entrée MetaC-App
import sys
import argparse
from pathlib import Path

# Ajout du répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from config.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="MetaC-App — Compagnon d'apprentissage adaptatif")
    parser.add_argument("--debug", action="store_true", help="Activer les logs DEBUG")
    parser.add_argument("pdf", nargs="?", help="Ouvrir directement un fichier PDF")
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    import logging
    logger = logging.getLogger("main")
    logger.info("Démarrage MetaC-App v1.3")

    from ui.app import NWoLApp
    app = NWoLApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)

    # Ouverture directe si un fichier est passé en argument
    if args.pdf:
        pdf_path = str(Path(args.pdf).resolve())
        app.after(200, lambda: app.open_pdf_path(pdf_path))

    app.mainloop()


if __name__ == "__main__":
    main()
