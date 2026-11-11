# Fonts

Self-hosted web fonts for the documentation site (no third-party requests at view time).
All are licensed under the SIL Open Font License 1.1; the licence texts are alongside.

| Family | Files | Source package |
|---|---|---|
| IBM Plex Sans (400, 400 italic, 500, 600; Latin and Greek subsets) | `ibm-plex-sans-*.woff2` | `@fontsource/ibm-plex-sans` 5.2.8 |
| IBM Plex Sans Condensed (500, 600, 700; Latin) | `ibm-plex-sans-condensed-*.woff2` | `@fontsource/ibm-plex-sans-condensed` 5.2.8 |
| IBM Plex Mono (400, 500; Latin) | `ibm-plex-mono-*.woff2` | `@fontsource/ibm-plex-mono` 5.2.7 |
| Source Serif 4 (variable weight, roman and italic; Latin and Greek subsets) | `source-serif-4-*.woff2` | `@fontsource-variable/source-serif-4` 5.2.9 |

To drop the fonts, delete this directory and the `@font-face` rules at the top of
`assets/css/site.css`; every family has a system fallback stack.
