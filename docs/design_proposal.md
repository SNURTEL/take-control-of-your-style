# Design proposal

## Funkcjonalność projektu

Celem projektu jest analiza równowagi między zachowaniem oryginalnej treści obrazu a wprowadzeniem do niego nowego stylu, przy użyciu różnych algorytmów przetwarzania obrazu. Badanie to pozwoli na głębsze zrozumienie, jak różne podejścia wpływają na końcowy efekt wizualny, balansując między wiernością reprezentacji zawartości a estetyką stylu (*style/content trade-off*).

## Harmonogram

### 18-24.03

- Podstawowy setup środowiska (struktura repo, tensorboard, conda + poetry)
- Zapoznanie się z tematem

### 25-31.03

- Rezerwa na Wielkanoc

### 1-7.04

- Hello world z transferu stylu

### 8-14.04

- Porównywanie różnych sieci jako ekstraktorów cech

### 14.04 - Raport z prac

### 15-21.04

- Porównywanie różnych sieci jako ekstraktorów cech ciąg dalszy

### 22-28.04

- Eksperymenty z kompromisem jakość reprezentacji - jakość stylu

### 29.04-5.05

- Eksperymenty z kompromisem jakość reprezentacji - jakość stylu ciąg dalszy

### 6-12.05

- Porządkowanie kodu bibliotecznego i serializacja modeli

### 13-19.05

- Tworzenie szkicu artykułu

### 20-26.05

- Tworzenie szkicu artykułu
- Ewentualna rezerwa

## Bibliografia

- https://arxiv.org/abs/1508.06576 - first net for style transfer
- https://arxiv.org/abs/1703.10593 - cycleGAN
- https://arxiv.org/abs/1703.06868 - adaIN layer
- https://arxiv.org/abs/2304.03198 - spacial attention

## Zakres eksperymentów

- Eksperymenty z optymalizacją parametrów funkcji straty, by zbalansować wpływ zachowania treści i adaptacji stylu, co może prowadzić do uzyskania bardziej satysfakcjonujących wyników wizualnych.
- Badanie efektywności mniejszych modeli w kontekście transferu stylu, skupiając się na ich zdolności do generalizacji i zachowania kluczowych cech stylistycznych przy jednoczesnym ograniczeniu szczegółowości. To podejście może oferować szybsze przetwarzanie i mniejsze wymagania obliczeniowe, otwierając drogę do efektywniejszych aplikacji w czasie rzeczywistym.
- Analiza wpływu zastosowania warstwy adaptacyjnej normalizacji instancji na dynamikę zmiany stylu, pozwalającej na bardziej płynne i kontrolowane przejścia estetyczne między zawartością a stylem.
- Ocena roli mechanizmów uwagi przestrzennej w kontekście poprawy selektywności modelu względem obszarów obrazu, które powinny być bardziej lub mniej podatne na modyfikacje stylistyczne.

^z uwagą, że minimialny ustalony zakres to ekeperymenty 1 i 2.

## Stack technologiczny
- środowisko: conda, poetry, ruff
- pytorch, lightning
- numpy
- matplotlib
- pomocnicze: tensorboard, github actions (CI)
