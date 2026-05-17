"""
src/forms/scrape_forms.py
==========================
Télécharge tous les formulaires PDF depuis les pages CNRA et RCAR.

Stratégie en 2 passes :
  1. Catalogue statique  — les URLs connues extraites des markdowns existants (fiable, rapide)
  2. Scraping live       — visite chaque page formulaire pour capturer les nouveaux PDFs
                          (nécessite accès internet ; les PDFs déjà téléchargés sont skippés)

Usage :
    python src/forms/scrape_forms.py                    # télécharge tout
    python src/forms/scrape_forms.py --static-only      # catalogue statique seulement
    python src/forms/scrape_forms.py --live-only        # scraping live seulement
    python src/forms/scrape_forms.py --dry-run          # liste sans télécharger

Sortie :
    data/forms/pdfs/          ← fichiers PDF téléchargés
    data/forms/forms.json     ← catalogue complet avec métadonnées
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

from loguru import logger

src_root = Path(__file__).resolve().parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from config.logger import setup_logger
    from config.settings import BASE_DIR, LOGS_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[2]
    LOGS_DIR = BASE_DIR / "logs"
    setup_logger = None

FORMS_DIR   = BASE_DIR / "data" / "forms"
PDFS_DIR    = FORMS_DIR / "pdfs"
CATALOG_FILE = FORMS_DIR / "forms.json"

CNRA_BASE = "https://www.cnra.ma"
RCAR_BASE = "https://www.rcar.ma"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9",
}

# ─────────────────────────────────────────────────────────────────────────────
# Catalogue statique — extrait des markdowns existants du projet
# Complété avec les pages CNRA dont les markdowns ne sont pas dans le zip
# ─────────────────────────────────────────────────────────────────────────────

STATIC_CATALOG: list[dict] = [
    # ══ CNRA — Imprimés CRAC ══════════════════════════════════════════════════
    {
        "form_id":    "cnra_crac_adhesion",
        "org":        "CNRA",
        "category":   "Imprimés CRAC",
        "category_slug": "I_CRAC",
        "title":      "Demande d'adhésion CRAC",
        "description": "Formulaire de demande d'adhésion au régime CRAC.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_CRAC",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/CRAC_adhesion.pdf",
        "filename":   "cnra_crac_adhesion.pdf",
    },
    # ══ CNRA — Imprimés Rentes AC ═════════════════════════════════════════════
    {
        "form_id":    "cnra_rac_avocat",
        "org":        "CNRA",
        "category":   "Imprimés Rentes AC",
        "category_slug": "I_RAC",
        "title":      "Engagement des frais d'avocat (AC)",
        "description": "Formulaire d'engagement des frais d'avocat pour rentes accidents de circulation.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RAC",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/Engagement_des_Frais_Avocat_AC.pdf",
        "filename":   "cnra_rac_avocat.pdf",
    },
    # ══ CNRA — Imprimés Rentes AT ═════════════════════════════════════════════
    {
        "form_id":    "cnra_rat_controle_vie",
        "org":        "CNRA",
        "category":   "Imprimés Rentes AT",
        "category_slug": "I_RAT",
        "title":      "Attestation sur l'honneur valant contrôle de vie",
        "description": "Attestation sur l'honneur valant contrôle de vie pour rentes accidents du travail.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RAT",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/Attestation_sur_honneur_valant_contrôle_de_vie.pdf",
        "filename":   "cnra_rat_controle_vie.pdf",
    },
    {
        "form_id":    "cnra_rat_majoration",
        "org":        "CNRA",
        "category":   "Imprimés Rentes AT",
        "category_slug": "I_RAT",
        "title":      "Demande de majoration AT",
        "description": "Formulaire de demande de majoration de rente accidents du travail.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RAT",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/demande de majoration َ AT 09092024.pdf",
        "filename":   "cnra_rat_majoration.pdf",
    },
    {
        "form_id":    "cnra_rat_attestation_majoration",
        "org":        "CNRA",
        "category":   "Imprimés Rentes AT",
        "category_slug": "I_RAT",
        "title":      "Demande d'attestation de majoration de rente",
        "description": "Formulaire de demande d'attestation de majoration de rente AT.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RAT",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/DEMANDE ATTESTATION DE MAJORATION DE RENTE 23092024.pdf",
        "filename":   "cnra_rat_attestation_majoration.pdf",
    },
    {
        "form_id":    "cnra_rat_non_remariage",
        "org":        "CNRA",
        "category":   "Imprimés Rentes AT",
        "category_slug": "I_RAT",
        "title":      "Attestation sur l'honneur pour non remariage",
        "description": "Attestation sur l'honneur de non remariage pour maintien de rente.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RAT",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/Attestation_sur_honneur_pour_non_remariage.pdf",
        "filename":   "cnra_rat_non_remariage.pdf",
    },
    # ══ CNRA — Imprimés FRAM ══════════════════════════════════════════════════
    {
        "form_id":    "cnra_fram_adhesion",
        "org":        "CNRA",
        "category":   "Imprimés FRAM",
        "category_slug": "I_FRAM",
        "title":      "FRAM — Demande d'adhésion/affiliation",
        "description": "Formulaire de demande d'adhésion ou d'affiliation au régime FRAM.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_FRAM",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/fram  Daffiliation 09092024.pdf",
        "filename":   "cnra_fram_adhesion.pdf",
    },
    {
        "form_id":    "cnra_fram_pension_deces",
        "org":        "CNRA",
        "category":   "Imprimés FRAM",
        "category_slug": "I_FRAM",
        "title":      "FRAM — Demande pension décès",
        "description": "Formulaire de demande de pension décès FRAM.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_FRAM",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/FRAM DPension décès 09092024.pdf",
        "filename":   "cnra_fram_pension_deces.pdf",
    },
    {
        "form_id":    "cnra_fram_pension",
        "org":        "CNRA",
        "category":   "Imprimés FRAM",
        "category_slug": "I_FRAM",
        "title":      "FRAM — Demande de pension",
        "description": "Formulaire de demande de pension FRAM.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_FRAM",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/FRAM-Dpension_05062023.pdf",
        "filename":   "cnra_fram_pension.pdf",
    },
    {
        "form_id":    "cnra_fram_rectification",
        "org":        "CNRA",
        "category":   "Imprimés FRAM",
        "category_slug": "I_FRAM",
        "title":      "FRAM — Demande de rectification",
        "description": "Formulaire de demande de rectification FRAM.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_FRAM",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/fram drectification 09092024.pdf",
        "filename":   "cnra_fram_rectification.pdf",
    },
    {
        "form_id":    "cnra_fram_pension_rcar",
        "org":        "CNRA",
        "category":   "Imprimés FRAM",
        "category_slug": "I_FRAM",
        "title":      "Demande de pension FRAM (RCAR)",
        "description": "Formulaire de demande de pension FRAM version 2025.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_FRAM",
        "pdf_url":    "https://www.rcar.ma/fr/services/commun/document-reference/Dpension-FRAM_2025.pdf",
        "filename":   "cnra_fram_pension_rcar_2025.pdf",
    },
    # ══ CNRA — Imprimés RECORE ════════════════════════════════════════════════
    {
        "form_id":    "cnra_recore_pension_capital",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Demande de rente et/ou capital",
        "description": "Formulaire de demande de rente et/ou capital retraite RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_DPension_retraite capital_10102023.pdf",
        "filename":   "cnra_recore_pension_capital.pdf",
    },
    {
        "form_id":    "cnra_recore_affiliation_individuelle",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Affiliation individuelle",
        "description": "Formulaire d'affiliation individuelle RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_Affiliation individuelle_04102023.pdf",
        "filename":   "cnra_recore_affiliation_individuelle.pdf",
    },
    {
        "form_id":    "cnra_recore_affiliation_groupe",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Affiliation individuelle / Adhésion groupe",
        "description": "Formulaire d'affiliation individuelle et adhésion groupe RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_Affiliation_individuelle_Groupe_04102023.pdf",
        "filename":   "cnra_recore_affiliation_groupe.pdf",
    },
    {
        "form_id":    "cnra_recore_deces_reversion",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Demande de rente/capital décès ou réversion",
        "description": "Formulaire de demande de liquidation décès ou réversion RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_DLiquidationDécès_09112023.pdf",
        "filename":   "cnra_recore_deces_reversion.pdf",
    },
    {
        "form_id":    "cnra_recore_rachat",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Demande de rachat total ou partiel",
        "description": "Formulaire de demande de rachat total ou partiel RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_DRachat total_ou_partiel_10102023.pdf",
        "filename":   "cnra_recore_rachat.pdf",
    },
    {
        "form_id":    "cnra_recore_invalidite",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Demande de liquidation en cas d'invalidité",
        "description": "Formulaire de demande de liquidation en cas d'invalidité RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_DLiqui_ation invalidité_10102023.pdf",
        "filename":   "cnra_recore_invalidite.pdf",
    },
    {
        "form_id":    "cnra_recore_versement_exceptionnel",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "RECORE — Versement exceptionnel",
        "description": "Formulaire de versement exceptionnel RECORE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_Versement_exceptionnel_30112023.pdf",
        "filename":   "cnra_recore_versement_exceptionnel.pdf",
    },
    {
        "form_id":    "cnra_recore_eservices",
        "org":        "CNRA",
        "category":   "Imprimés RECORE",
        "category_slug": "I_RECOR",
        "title":      "Demande d'utilisation des E-services CNRA",
        "description": "Formulaire de demande d'accès aux E-services CNRA.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_RECOR",
        "pdf_url":    "https://www.rcar.ma/fr/services/commun/document-reference/Demande_d'utilisation_des_E-services_VF_-_CNRA.pdf",
        "filename":   "cnra_recore_eservices.pdf",
    },
    # ══ CNRA — Imprimés MRE ═══════════════════════════════════════════════════
    {
        "form_id":    "cnra_mre_rente_capital",
        "org":        "CNRA",
        "category":   "Imprimés MRE",
        "category_slug": "I_MRE",
        "title":      "MRE — Demande de rente et/ou capital",
        "description": "Formulaire de demande de rente et/ou capital pour Marocains Résidant à l'Étranger.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_MRE",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_DPension_retraite capital_10102023.pdf",
        "filename":   "cnra_mre_rente_capital.pdf",
    },
    {
        "form_id":    "cnra_mre_affiliation_individuelle",
        "org":        "CNRA",
        "category":   "Imprimés MRE",
        "category_slug": "I_MRE",
        "title":      "MRE — Affiliation individuelle",
        "description": "Formulaire d'affiliation individuelle MRE.",
        "page_url":   f"{CNRA_BASE}/formulaire-piece/I_MRE",
        "pdf_url":    f"{CNRA_BASE}/images/galerie/docs/RECORE_Affiliation individuelle_04102023.pdf",
        "filename":   "cnra_mre_affiliation_individuelle.pdf",
    },
    # ══ RCAR — Imprimés et formulaires ════════════════════════════════════════
    {
        "form_id":    "rcar_justificatif_versement",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Justificatif de versement (VR1)",
        "description": "Formulaire de justificatif de versement RCAR — Imprimé VR1.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/Justificatif de versement.pdf",
        "filename":   "rcar_justificatif_versement.pdf",
    },
    {
        "form_id":    "rcar_pension_retraite",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Formulaire pension de retraite — Demande d'allocation de retraite",
        "description": "Formulaire de demande d'allocation de retraite RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/DEMANDE D'ALLOCATION DE RETRAITE.pdf",
        "filename":   "rcar_pension_retraite.pdf",
    },
    {
        "form_id":    "rcar_pension_deces_reversion",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Formulaire pension de décès et de réversion",
        "description": "Formulaire de demande d'allocation de décès RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/DEMANDE D'ALLOCATION DE DECES.pdf",
        "filename":   "rcar_pension_deces_reversion.pdf",
    },
    {
        "form_id":    "rcar_pension_invalidite",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Formulaire pension d'invalidité",
        "description": "Formulaire de demande d'allocation d'invalidité RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/DEMANDE_ALLOCATION_INVALIDITE.pdf",
        "filename":   "rcar_pension_invalidite.pdf",
    },
    {
        "form_id":    "rcar_adhesion_regime_general",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Déclaration d'adhésion au régime général",
        "description": "Formulaire d'adhésion au régime général RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/declaration-adhesion.pdf",
        "filename":   "rcar_adhesion_regime_general.pdf",
    },
    {
        "form_id":    "rcar_convention_regime_complementaire",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Convention d'adhésion au régime complémentaire",
        "description": "Modèle de convention d'adhésion par l'employeur au régime complémentaire RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/Convention_D_Adhesion_Au_Regime_Complementaire.pdf",
        "filename":   "rcar_convention_regime_complementaire.pdf",
    },
    {
        "form_id":    "rcar_validation_anterieur_rg",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Validation des services antérieurs — Régime Général",
        "description": "Formulaire de demande de validation des services antérieurs (Régime Général) RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/validation_service_anterieur_RG.pdf",
        "filename":   "rcar_validation_anterieur_rg.pdf",
    },
    {
        "form_id":    "rcar_validation_anterieur_rc",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Validation des services antérieurs — Régime Complémentaire",
        "description": "Formulaire de demande de validation des services antérieurs (Régime Complémentaire) RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/validation_service_anterieurRC.pdf",
        "filename":   "rcar_validation_anterieur_rc.pdf",
    },
    {
        "form_id":    "rcar_declaration_cotisations_vsa",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Déclaration cotisations VSA et rachat",
        "description": "Support pour déclarer les cotisations de validation des services antérieurs et rachat RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/Déclaration des cotisations validation et rachat RG + RC.pdf",
        "filename":   "rcar_declaration_cotisations_vsa.pdf",
    },
    {
        "form_id":    "rcar_modification_salaires",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Modification de salaires",
        "description": "Support pour modifier les déclarations de salaires et cotisations transmises au RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/Modification des déclarations.pdf",
        "filename":   "rcar_modification_salaires.pdf",
    },
    {
        "form_id":    "rcar_validation_service_civil",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Validation des services antérieurs — Service Civil",
        "description": "Formulaire de demande de validation des services antérieurs (Service Civil) RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/demande-validation-SC-.pdf",
        "filename":   "rcar_validation_service_civil.pdf",
    },
    {
        "form_id":    "rcar_assurance_volontaire",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Souscription à l'assurance volontaire",
        "description": "Formulaire de demande de souscription à l'assurance volontaire RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/Demande souscription Assurance Volontaire.pdf",
        "filename":   "rcar_assurance_volontaire.pdf",
    },
    {
        "form_id":    "rcar_demande_rachat",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Demande de rachat de services",
        "description": "Formulaire de demande de rachat de période de services RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    f"{RCAR_BASE}/uploads/imprimes_depliants_guides/Demande_rachat_periode.pdf",
        "filename":   "rcar_demande_rachat.pdf",
    },
    {
        "form_id":    "rcar_eservices",
        "org":        "RCAR",
        "category":   "Imprimés et formulaires RCAR",
        "category_slug": "RCAR_IMPFM",
        "title":      "Demande d'utilisation des E-services RCAR",
        "description": "Formulaire de demande d'accès aux E-services RCAR.",
        "page_url":   f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM",
        "pdf_url":    "https://www.rcar.ma/fr/services/commun/document-reference/Demande_d'utilisation_des_E-services_VF_-_RCAR.pdf",
        "filename":   "rcar_eservices.pdf",
    },
]

# Pages à scraper pour découverte dynamique de nouveaux PDFs
LIVE_FORM_PAGES: list[dict] = [
    {"org": "CNRA", "category": "Imprimés CRAC",    "category_slug": "I_CRAC",  "url": f"{CNRA_BASE}/formulaire-piece/I_CRAC"},
    {"org": "CNRA", "category": "Imprimés Rentes AC","category_slug": "I_RAC",  "url": f"{CNRA_BASE}/formulaire-piece/I_RAC"},
    {"org": "CNRA", "category": "Imprimés Rentes AT","category_slug": "I_RAT",  "url": f"{CNRA_BASE}/formulaire-piece/I_RAT"},
    {"org": "CNRA", "category": "Imprimés FRAM",    "category_slug": "I_FRAM",  "url": f"{CNRA_BASE}/formulaire-piece/I_FRAM"},
    {"org": "CNRA", "category": "Imprimés RECORE",  "category_slug": "I_RECOR", "url": f"{CNRA_BASE}/formulaire-piece/I_RECOR"},
    {"org": "CNRA", "category": "Imprimés MRE",     "category_slug": "I_MRE",   "url": f"{CNRA_BASE}/formulaire-piece/I_MRE"},
    {"org": "CNRA", "category": "Imprimés CFC",     "category_slug": "I_CFC",   "url": f"{CNRA_BASE}/formulaire-piece/I_CFC"},
    {"org": "CNRA", "category": "Imprimés Douayer Zmane","category_slug": "DZM","url": f"{CNRA_BASE}/formulaire-piece/DZM"},
    {"org": "RCAR", "category": "Imprimés et formulaires RCAR","category_slug": "RCAR_IMPFM","url": f"{RCAR_BASE}/fr/formulaire-piece/RCAR_IMPFM"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Scraping live
# ─────────────────────────────────────────────────────────────────────────────

def _make_filename(title: str, org: str, category_slug: str) -> str:
    """Génère un nom de fichier propre depuis le titre."""
    clean = re.sub(r"[^\w\s-]", "", title.lower())
    clean = re.sub(r"[\s-]+", "_", clean).strip("_")[:60]
    return f"{org.lower()}_{category_slug.lower()}_{clean}.pdf"


def _make_form_id(title: str, org: str, category_slug: str) -> str:
    clean = re.sub(r"[^\w]", "_", f"{org}_{category_slug}_{title}").lower()[:80]
    return re.sub(r"_+", "_", clean).strip("_")


def scrape_page(page_info: dict) -> list[dict]:
    """
    Scrape une page de formulaires et retourne la liste des PDFs trouvés.
    Ne retourne que les PDFs pas encore dans le catalogue statique.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("httpx/beautifulsoup4 non installé — scraping live désactivé")
        return []

    org   = page_info["org"]
    cat   = page_info["category"]
    slug  = page_info["category_slug"]
    url   = page_info["url"]
    base  = CNRA_BASE if org == "CNRA" else RCAR_BASE

    forms = []
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Erreur scraping {}: {}", url, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Chercher les sections h4 + lien TELECHARGER
    for h4 in soup.find_all(["h4", "h3"]):
        raw_title = h4.get_text(strip=True).replace("*", "").strip()
        if not raw_title:
            continue

        # Chercher le lien "TELECHARGER" / "TÉLÉCHARGER" dans les éléments suivants
        next_el = h4.find_next_sibling()
        pdf_url = None
        while next_el and next_el.name not in ("h4", "h3"):
            a = next_el.find("a", href=True) if hasattr(next_el, "find") else None
            if a:
                href = a["href"].strip()
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(base, href)
                    break
            next_el = next_el.find_next_sibling() if hasattr(next_el, "find_next_sibling") else None

        if pdf_url:
            filename = _make_filename(raw_title, org, slug)
            forms.append({
                "form_id":       _make_form_id(raw_title, org, slug),
                "org":           org,
                "category":      cat,
                "category_slug": slug,
                "title":         raw_title,
                "description":   f"Formulaire {cat} — {raw_title}",
                "page_url":      url,
                "pdf_url":       pdf_url,
                "filename":      filename,
                "source":        "scraped",
            })

    logger.info("Scraping {}: {} PDFs trouvés", url, len(forms))
    return forms


# ─────────────────────────────────────────────────────────────────────────────
# Téléchargement des PDFs
# ─────────────────────────────────────────────────────────────────────────────

def download_pdf(form: dict, output_dir: Path, delay: float = 1.0) -> bool:
    """
    Télécharge un PDF et le sauvegarde dans output_dir.
    Retourne True si le fichier a été téléchargé (ou existait déjà).
    """
    try:
        import httpx
    except ImportError:
        logger.error("httpx non installé : pip install httpx")
        return False

    dest = output_dir / form["filename"]
    if dest.exists() and dest.stat().st_size > 500:
        logger.debug("Déjà téléchargé: {}", form["filename"])
        form["downloaded"] = True
        form["local_path"] = str(dest)
        return True

    pdf_url = form["pdf_url"]
    try:
        resp = httpx.get(pdf_url, headers=HEADERS, timeout=30, follow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            logger.warning("Contenu non-PDF pour {}: {}", form["filename"], content_type)

        dest.write_bytes(resp.content)
        form["downloaded"] = True
        form["local_path"] = str(dest)
        logger.info("Téléchargé: {} ({:.1f} KB)", form["filename"], len(resp.content) / 1024)
        time.sleep(delay)
        return True

    except Exception as exc:
        logger.warning("Erreur téléchargement {} — {}: {}", form["form_id"], pdf_url, exc)
        form["downloaded"] = False
        form["local_path"] = None
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run(
    static_only: bool = False,
    live_only:   bool = False,
    dry_run:     bool = False,
    delay:       float = 1.0,
) -> list[dict]:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    FORMS_DIR.mkdir(parents=True, exist_ok=True)

    # Charger le catalogue existant
    existing: dict[str, dict] = {}
    if CATALOG_FILE.exists():
        try:
            data = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
            existing = {f["form_id"]: f for f in data if isinstance(f, dict)}
            logger.info("Catalogue existant: {} entrées", len(existing))
        except Exception as exc:
            logger.warning("Catalogue illisible: {}", exc)

    all_forms: dict[str, dict] = {**existing}

    # Passe 1 : catalogue statique
    if not live_only:
        for form in STATIC_CATALOG:
            if form["form_id"] not in all_forms:
                all_forms[form["form_id"]] = {**form, "source": form.get("source", "static")}
        logger.info("Catalogue statique: {} formulaires", len(STATIC_CATALOG))

    # Passe 2 : scraping live
    if not static_only:
        logger.info("Scraping live des pages formulaires...")
        scraped_new = 0
        for page_info in LIVE_FORM_PAGES:
            found = scrape_page(page_info)
            for form in found:
                if form["form_id"] not in all_forms:
                    all_forms[form["form_id"]] = form
                    scraped_new += 1
            time.sleep(delay * 0.5)
        logger.info("Scraping live: {} nouveaux formulaires", scraped_new)

    forms_list = list(all_forms.values())
    pdf_only   = [f for f in forms_list if (f.get("pdf_url") or "").lower().endswith(".pdf")]

    logger.info("Total formulaires à traiter: {} ({} PDFs)", len(forms_list), len(pdf_only))

    if dry_run:
        print("\n── Dry-run : formulaires détectés ──")
        for f in forms_list:
            print(f"  [{f['org']:4}] {f['category_slug']:12} | {f['title'][:60]}")
            if f.get("pdf_url"):
                print(f"         → {f['pdf_url']}")
        CATALOG_FILE.write_text(json.dumps(forms_list, ensure_ascii=False, indent=2), encoding="utf-8")
        return forms_list

    # Téléchargement
    ok, fail = 0, 0
    for form in pdf_only:
        success = download_pdf(form, PDFS_DIR, delay=delay)
        if success:
            ok += 1
        else:
            fail += 1

    # Sauvegarde catalogue
    CATALOG_FILE.write_text(json.dumps(forms_list, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Terminé: {}/{} PDFs téléchargés ({} échecs) → {}",
        ok, len(pdf_only), fail, CATALOG_FILE
    )
    return forms_list


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Télécharge les formulaires CNRA/RCAR")
    p.add_argument("--static-only", action="store_true", help="Utiliser uniquement le catalogue statique")
    p.add_argument("--live-only",   action="store_true", help="Scraper uniquement les pages live")
    p.add_argument("--dry-run",     action="store_true", help="Lister sans télécharger")
    p.add_argument("--delay", type=float, default=1.0,  help="Délai entre requêtes (s)")
    return p.parse_args()


if __name__ == "__main__":
    if setup_logger is not None:
        setup_logger(log_dir=LOGS_DIR, source="fetch_forms")

    args = parse_args()
    results = run(
        static_only=args.static_only,
        live_only=args.live_only,
        dry_run=args.dry_run,
        delay=args.delay,
    )
    downloaded = sum(1 for f in results if f.get("downloaded"))
    print(f"\nRésultat: {downloaded}/{len(results)} formulaires téléchargés")
    print(f"Catalogue: {CATALOG_FILE}")
