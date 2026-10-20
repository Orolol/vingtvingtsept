"""Notices de la Commission des sondages (https://www.commission-des-sondages.fr).

Source live des sondages de la présidentielle 2027 et, pour certains instituts, des résultats
bruts. Commandes : `vvs ingest commission --since AAAA-MM-JJ` puis `vvs parse commission`.

Structure du site (relevée le 2026-10-20)
-----------------------------------------

- Application CakePHP sous `/notices/`. La page d'accueil `/notices/` ne liste que l'année en
  cours, en panneaux mensuels (`h2.notices-mois-titre`, puis des `p.download-line`). Les pages
  par année `/notices/medias/dossiers/view/<année>` répondent 500 (au moins pour 2025).
- Thèmes : `/notices/medias/fichiers/bytag/14/2027-Presidentielle` liste toutes les notices
  étiquetées « 2027 - Présidentielle » (190 environ, de septembre 2024 à aujourd'hui) sur une
  seule page, sans pagination, chaque ligne suffixée du mois de dépôt (« - Octobre 2026 »).
  L'étiquette n'est pas systématique : quelques notices « Pres » de 2026 n'y sont pas.
- Recherche : POST `/notices/medias/fichiers/search` avec `searchterm=Pres` renvoie toutes
  les notices dont le titre contient « pres » (insensible aux accents), 2016 à aujourd'hui,
  sur une seule page. On garde l'union « thème 2027 » + « titre portant le jeton Pres /
  Présidentielle / Presitrack » (voir `listing.is_presidential`).
- Une ligne de liste : `<a href="/notices/medias/fichiers/add/<file_id>">` avec pour texte le
  titre saisi par la Commission, de forme libre mais régulière :
  « <numéro> Pres [sujet] <INSTITUT> [commanditaire] <jour> <mois> » (ex. « 10325 Pres Focus
  droite et centre CLUSTER17 Le Point 18 octobre »). Le numéro (10325, parfois suffixé « A »
  pour une notice rectificative ou complémentaire) est celui de la Commission ; `file_id`
  (2341) est l'identifiant de fichier, croissant avec le dépôt. La date du titre est en
  général celle de publication ; l'année n'y figure pas, on la prend du mois de dépôt.
- `/notices/medias/fichiers/add/<file_id>` répond 302 vers le PDF
  `/notices/files/notices/<année>/<mois>/<slug>.pdf`. Toutes les notices 2025-2026 sont des
  PDF texte (aucun scan) ; un lien est mort (404 sur la notice 10251 A, `file_id` 2269).

Contenu des notices
-------------------

Le contenu dépend de l'institut, pas de la Commission : chaque institut dépose son propre
document. Métadonnées toujours présentes (article 2 et 3 de la loi de 1977) : institut,
commanditaire, échantillon, dates de terrain, mode, critères de redressement. Selon les cas :

- Ifop, Elabe, Toluna Harris Interactive, Ipsos BVA, Verian : résultats publiés (redressés)
  seulement ; Ifop a publié les colonnes « Résultats bruts » et chaque redressement pour sa
  grande enquête Hexagone (mai 2025), pas dans ses baromètres ;
- Cluster17 (dès octobre 2025), YouGov (septembre 2026), OpinionWay (dès mars 2026) :
  brut et redressé pour chaque hypothèse ;
- Odoxa : notice réduite aux articles 2 et 3 et au questionnaire, sans résultats.

`listing` télécharge, `notice` lit un PDF (texte, métadonnées, tableaux), `aliases` porte les
tables institut/candidat, `parse` produit les parquet curated.
"""
