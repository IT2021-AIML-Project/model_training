<div align="center">

# 🦺 PPE Detection System

### Real-Time Personal Protective Equipment Detection with YOLOv8n

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-00BFFF?logo=github)
![mAP](https://img.shields.io/badge/mAP%400.5-0.837-brightgreen)
![Latency](https://img.shields.io/badge/Latency-22ms-orange)
![GPU](https://img.shields.io/badge/GPU-GTX%201650-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

*Two-phase curriculum training for construction-site PPE compliance monitoring*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Training Pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Training](#training)
  - [Phase 1 — Roboflow](#phase-1--roboflow-pre-training)
  - [Phase 2 — Client Fine-Tuning](#phase-2--client-data-fine-tuning)
- [Results](#results)
  - [Final Evaluation](#final-evaluation)
  - [Per-Class Metrics](#per-class-metrics)
  - [Confusion Matrix](#confusion-matrix)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inference & Deployment](#inference--deployment)
- [Repository Structure](#repository-structure)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This repository implements a **real-time PPE detection system** for construction and industrial environments. The system detects 7 PPE-compliance classes across live video streams using **YOLOv8n** — chosen for its exceptional inference speed (22 ms on GTX 1650) while maintaining a strong mAP@0.5 of **0.837**.

Training follows a **two-phase curriculum**: Phase 1 bootstraps on a broad Roboflow PPE dataset; Phase 2 fine-tunes on client-specific imagery for domain adaptation.

### Key Numbers

| Metric | Value |
|---|---|
| Model | YOLOv8n |
| Parameters | 3.2 M |
| Inference latency | **22 ms** (GTX 1650) |
| Training epochs | 50 × 2 phases |
| mAP@0.5 | **0.837** |
| mAP@0.5-0.95 | ~0.530 |
| Best F1 | **0.83** @ conf 0.380 |
| Max precision | 1.00 @ conf 0.938 |
| Classes | 7 |

---

## Training Pipeline

The diagram below shows the full end-to-end pipeline from raw data to deployment:


<svg width="100%" viewBox="0 0 680 900" role="img" style="" xmlns="http://www.w3.org/2000/svg">
  <title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">YOLOv8n PPE &amp; Vehicle Detection Training Pipeline</title>
  <desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">End-to-end training pipeline: Roboflow dataset → preprocessing → model selection → fine-tuning on client data → evaluation → deployment</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  <mask id="imagine-text-gaps-o5lrm8" maskUnits="userSpaceOnUse"><rect x="0" y="0" width="680" height="900" fill="white"/><rect x="298.3937683105469" y="43.79999923706055" width="83.771484375" height="22.399999618530273" fill="black" rx="2"/><rect x="166.1374969482422" y="64.4000015258789" width="348.751708984375" height="19.199999809265137" fill="black" rx="2"/><rect x="291.5500183105469" y="133.8000030517578" width="96.9000015258789" height="22.399999618530273" fill="black" rx="2"/><rect x="201.65000915527344" y="154.40000915527344" width="277.27569580078125" height="19.199999809265137" fill="black" rx="2"/><rect x="285.4562683105469" y="222.8000030517578" width="109.0875015258789" height="22.399999618530273" fill="black" rx="2"/><rect x="142.99375915527344" y="242.40000915527344" width="395.0416564941406" height="19.199999809265137" fill="black" rx="2"/><rect x="123.5" y="272" width="43.06328201293945" height="20" fill="black" rx="2"/><rect x="256.2749938964844" y="272" width="47.86562728881836" height="20" fill="black" rx="2"/><rect x="370.625" y="272" width="58.767189025878906" height="20" fill="black" rx="2"/><rect x="499.3125" y="272" width="61.375" height="20" fill="black" rx="2"/><rect x="107.07500457763672" y="295.3999938964844" width="75.44393157958984" height="19.199999809265137" fill="black" rx="2"/><rect x="261.32501220703125" y="295.3999938964844" width="36.55000114440918" height="19.199999809265137" fill="black" rx="2"/><rect x="375.6125183105469" y="295.3999938964844" width="49.807960510253906" height="19.199999809265137" fill="black" rx="2"/><rect x="511.1812744140625" y="295.3999938964844" width="38.44394493103027" height="19.199999809265137" fill="black" rx="2"/><rect x="113.8499984741211" y="321.3999938964844" width="62.53594970703125" height="19.199999809265137" fill="black" rx="2"/><rect x="260.2124938964844" y="321.3999938964844" width="38.77499961853027" height="19.199999809265137" fill="black" rx="2"/><rect x="373.5062561035156" y="321.3999938964844" width="54.01992416381836" height="19.199999809265137" fill="black" rx="2"/><rect x="510.625" y="321.3999938964844" width="39.26596641540527" height="19.199999809265137" fill="black" rx="2"/><rect x="113.80000305175781" y="347.3999938964844" width="62.6319580078125" height="19.199999809265137" fill="black" rx="2"/><rect x="261.46875" y="347.3999938964844" width="36.26250076293945" height="19.199999809265137" fill="black" rx="2"/><rect x="374.7124938964844" y="347.3999938964844" width="51.60195541381836" height="19.199999809265137" fill="black" rx="2"/><rect x="510.54998779296875" y="347.3999938964844" width="38.89999961853027" height="19.199999809265137" fill="black" rx="2"/><rect x="112.7562484741211" y="361.3999938964844" width="64.71393966674805" height="19.199999809265137" fill="black" rx="2"/><rect x="261.0625" y="361.3999938964844" width="37.07500076293945" height="19.199999809265137" fill="black" rx="2"/><rect x="375.54376220703125" y="361.3999938964844" width="49.94599533081055" height="19.199999809265137" fill="black" rx="2"/><rect x="510.54998779296875" y="361.3999938964844" width="38.89999961853027" height="19.199999809265137" fill="black" rx="2"/><rect x="215.05625915527344" y="421.8000183105469" width="250.4384765625" height="22.399999618530273" fill="black" rx="2"/><rect x="201.60000610351562" y="442.3999938964844" width="276.4417419433594" height="19.199999809265137" fill="black" rx="2"/><rect x="263.28125" y="511.79998779296875" width="153.4375" height="22.399999618530273" fill="black" rx="2"/><rect x="87.3187484741211" y="532.4000244140625" width="505.1335144042969" height="19.199999809265137" fill="black" rx="2"/><rect x="236.3249969482422" y="601.7999877929688" width="207.912109375" height="22.399999618530273" fill="black" rx="2"/><rect x="167.8625030517578" y="622.4000244140625" width="343.9176940917969" height="19.199999809265137" fill="black" rx="2"/><rect x="303.09375" y="691.7999877929688" width="73.8125" height="22.399999618530273" fill="black" rx="2"/><rect x="161.5812530517578" y="712.4000244140625" width="357.1876525878906" height="19.199999809265137" fill="black" rx="2"/><rect x="284.5687561035156" y="781.7999877929688" width="111.7373046875" height="22.399999618530273" fill="black" rx="2"/><rect x="220.96250915527344" y="802.4000244140625" width="238.0749969482422" height="19.199999809265137" fill="black" rx="2"/><rect x="54" y="853.4000244140625" width="35.26397705078125" height="19.199999809265137" fill="black" rx="2"/><rect x="123.20000457763672" y="853.4000244140625" width="53.469970703125" height="19.199999809265137" fill="black" rx="2"/><rect x="211.1999969482422" y="853.4000244140625" width="93.54595184326172" height="19.199999809265137" fill="black" rx="2"/><rect x="334" y="853.4000244140625" width="70.63393783569336" height="19.199999809265137" fill="black" rx="2"/><rect x="426" y="853.4000244140625" width="77.80596160888672" height="19.199999809265137" fill="black" rx="2"/></mask></defs>

  <!-- ── STAGE 1: Data Source ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="30" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="55" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Data source</text>
    <text x="340" y="74" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Roboflow public dataset · 25-class PPE &amp; vehicle annotations</text>
  </g>

  <!-- arrow -->
  <line x1="340" y1="94" x2="340" y2="118" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 2: Preprocessing ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="120" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="145" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Preprocessing</text>
    <text x="340" y="164" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Resize · augment · normalise · YOLO label format</text>
  </g>

  <!-- arrow -->
  <line x1="340" y1="184" x2="340" y2="208" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 3: Model Selection (box with inner table) ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="210" width="600" height="170" rx="10" stroke-width="0.5" style="fill:rgb(60, 52, 137);stroke:rgb(175, 169, 236);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="234" text-anchor="middle" dominant-baseline="central" style="fill:rgb(206, 203, 246);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Model selection</text>
    <text x="340" y="252" text-anchor="middle" dominant-baseline="central" style="fill:rgb(175, 169, 236);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Benchmark on NVIDIA GeForce GTX 1650 · COCO pretrained weights</text>
  </g>

  <!-- Table header -->
  <rect x="60" y="265" width="560" height="26" rx="4" fill="none" stroke="none" style="fill:none;stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="145" y="282" text-anchor="middle" dominant-baseline="central" style="font-weight:500;fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:500;text-anchor:middle;dominant-baseline:central">Model</text>
  <text x="280" y="282" text-anchor="middle" dominant-baseline="central" style="font-weight:500;fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:500;text-anchor:middle;dominant-baseline:central">Params</text>
  <text x="400" y="282" text-anchor="middle" dominant-baseline="central" style="font-weight:500;fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:500;text-anchor:middle;dominant-baseline:central">Inference</text>
  <text x="530" y="282" text-anchor="middle" dominant-baseline="central" style="font-weight:500;fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:500;text-anchor:middle;dominant-baseline:central">mAP@0.5</text>
  <line x1="60" y1="291" x2="620" y2="291" stroke="var(--color-border-tertiary)" stroke-width="0.5" mask="url(#imagine-text-gaps-o5lrm8)" style="fill:rgb(0, 0, 0);stroke:rgba(222, 220, 209, 0.15);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- Row 1 — selected -->
  <rect x="60" y="292" width="560" height="26" rx="3" opacity="0.35" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:0.35;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="145" y="305" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">YOLOv8n ★</text>
  <text x="280" y="305" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">3.2M</text>
  <text x="400" y="305" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">~22 ms</text>
  <text x="530" y="305" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">0.837</text>
  <!-- Row 2 -->
  <text x="145" y="331" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">YOLOv8s</text>
  <text x="280" y="331" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">11.2M</text>
  <text x="400" y="331" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">~128 ms</text>
  <text x="530" y="331" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">0.943</text>
  <!-- Row 3 -->
  <text x="145" y="357" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">YOLOv9s</text>
  <text x="280" y="357" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">7.2M</text>
  <text x="400" y="357" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">~116 ms</text>
  <text x="530" y="357" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">0.950</text>
  <!-- Row 4 -->
  <text x="145" y="371" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">YOLOv11s</text>
  <text x="280" y="371" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">9.4M</text>
  <text x="400" y="371" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">~90 ms</text>
  <text x="530" y="371" text-anchor="middle" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">0.950</text>

  <line x1="60" y1="383" x2="620" y2="383" stroke="var(--color-border-tertiary)" stroke-width="0.5" style="fill:rgb(0, 0, 0);stroke:rgba(222, 220, 209, 0.15);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- arrow -->
  <line x1="340" y1="381" x2="340" y2="406" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 4: Phase 1 Training ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="408" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(12, 68, 124);stroke:rgb(133, 183, 235);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="433" text-anchor="middle" dominant-baseline="central" style="fill:rgb(181, 212, 244);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Phase 1 — Roboflow dataset fine-tune</text>
    <text x="340" y="452" text-anchor="middle" dominant-baseline="central" style="fill:rgb(133, 183, 235);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">50 epochs · GTX 1650 · mAP@0.5 = 0.838 → 0.865</text>
  </g>

  <!-- arrow -->
  <line x1="340" y1="472" x2="340" y2="496" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 5: Client Dataset ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="498" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(113, 43, 19);stroke:rgb(240, 153, 123);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="523" text-anchor="middle" dominant-baseline="central" style="fill:rgb(245, 196, 179);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Client dataset labelling</text>
    <text x="340" y="542" text-anchor="middle" dominant-baseline="central" style="fill:rgb(240, 153, 123);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">7 classes: Hardhat · Mask · NO-Hardhat · NO-Mask · NO-Safety Vest · Person · Safety Vest</text>
  </g>

  <!-- arrow -->
  <line x1="340" y1="562" x2="340" y2="586" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 6: Phase 2 Training ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="588" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(12, 68, 124);stroke:rgb(133, 183, 235);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="613" text-anchor="middle" dominant-baseline="central" style="fill:rgb(181, 212, 244);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Phase 2 — client data fine-tune</text>
    <text x="340" y="632" text-anchor="middle" dominant-baseline="central" style="fill:rgb(133, 183, 235);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">50 epochs · weights from Phase 1 · mAP@0.5 = 0.594 → 0.865</text>
  </g>

  <!-- arrow -->
  <line x1="340" y1="652" x2="340" y2="676" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 7: Evaluation ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="678" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(60, 52, 137);stroke:rgb(175, 169, 236);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="703" text-anchor="middle" dominant-baseline="central" style="fill:rgb(206, 203, 246);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Evaluation</text>
    <text x="340" y="722" text-anchor="middle" dominant-baseline="central" style="fill:rgb(175, 169, 236);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Precision · Recall · mAP@0.5 · mAP@0.5-0.95 · confusion matrix</text>
  </g>

  <!-- arrow -->
  <line x1="340" y1="742" x2="340" y2="766" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- ── STAGE 8: Export / Deploy ── -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="40" y="768" width="600" height="64" rx="10" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="793" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Export &amp; deploy</text>
    <text x="340" y="812" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">best.pt · ONNX / TensorRT · inference API</text>
  </g>

  <!-- Legend -->
  <rect x="40" y="856" width="12" height="12" rx="2" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="58" y="863" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">Data</text>
  <rect x="110" y="856" width="12" height="12" rx="2" stroke-width="0.5" style="fill:rgb(12, 68, 124);stroke:rgb(133, 183, 235);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="128" y="863" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">Training</text>
  <rect x="198" y="856" width="12" height="12" rx="2" stroke-width="0.5" style="fill:rgb(60, 52, 137);stroke:rgb(175, 169, 236);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="216" y="863" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">Selection / eval</text>
  <rect x="320" y="856" width="12" height="12" rx="2" stroke-width="0.5" style="fill:rgb(113, 43, 19);stroke:rgb(240, 153, 123);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="338" y="863" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">Client data</text>
  <rect x="412" y="856" width="12" height="12" rx="2" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="430" y="863" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">Deployment</text>
</svg>

---

## Dataset

Training data combines a public Roboflow PPE dataset (Phase 1) with proprietary client imagery (Phase 2). All images are resized to **640×640** pixels with YOLO-format annotations.

### Class Distribution

![Dataset Labels](plots/labels.jpg)

| Class | Train Instances | Notes |
|---|---|---|
| Person | 9,691 | Dominant class (~37% of all annotations) |
| NO-Safety Vest | 3,957 | Most common violation |
| Hardhat | 3,362 | Core positive PPE class |
| NO-Mask | 3,209 | Violation class |
| Safety Vest | 3,170 | Positive PPE class |
| NO-Hardhat | 2,318 | Fewer examples — hardest class |
| Mask | 1,743 | Smallest class — consider augmentation |

> **Note:** The bounding-box distribution (top-right) shows objects span a wide range of image positions and scales. The width/height scatter (bottom-right) confirms most PPE items are small-to-medium relative to frame size.

---

## Model Selection

Four YOLO variants were benchmarked on the **NVIDIA GeForce GTX 1650** across parameter count, inference latency, and mAP@0.5. YOLOv8n was selected for its 5× speed advantage over YOLOv8s with an acceptable accuracy trade-off.

![Model Comparison](plots/model_comparison.png)

| Model | Params (M) | Latency (ms) | mAP@0.5 | Selected |
|---|---|---|---|---|
| **YOLOv8n** | **3.2** | **22** | 0.837 | ✅ |
| YOLOv8s | 11.2 | 128 | 0.943 | — |
| YOLOv9s | 7.2 | 116 | 0.950 | — |
| YOLOv11s | 9.4 | 90 | 0.950 | — |

---

## Training

### Hyperparameters

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Image size | 640×640 | Optimizer | SGD |
| Batch size | 16 | Initial LR | 0.00088 |
| Epochs (per phase) | 50 | LR schedule | Cosine decay |
| Warmup epochs | 3 | Momentum | 0.937 |
| Weight decay | 0.0005 | IOU threshold | 0.7 |

---

### Phase 1 — Roboflow Pre-Training

Phase 1 trains from the COCO-pretrained YOLOv8n backbone on the full Roboflow PPE dataset. This establishes task-specific feature representations before domain adaptation.

#### Loss Curves

All three YOLO loss components (box localisation, classification, DFL) converge smoothly across 50 epochs with no instability.

![Loss Curves](plots/loss_curvesRoboflow.png)

![Total Loss](plots/total_lossRoboflow.png)

#### Learning Rate Schedule

Linear warmup for 3 epochs → cosine decay to near-zero by epoch 50.

![LR Schedule](plots/lr_curveRoboflow.png)

#### mAP Performance

![mAP Curves](plots/map_curvesRoboflow.png)

The gap between mAP@0.5 and mAP@0.5-0.95 narrows from ~0.37 → ~0.31 over training, indicating improving bounding-box localisation quality.

![mAP Gap](plots/map_gapRoboflow.png)

#### Precision & Recall

![Precision Recall](plots/precision_recallRoboflow.png)

![Precision Recall Scatter](plots/precision_recall_scatterRoboflow.png)

*Scatter plot coloured by epoch — the trajectory moves toward high precision and high recall over training.*

#### F1 Score

![F1 Curve](plots/f1_curveRoboflow.png)

---

### Phase 2 — Client Data Fine-Tuning

Phase 2 loads the Phase 1 best checkpoint and fine-tunes on client-specific imagery, adapting to the target deployment environment.

**Phase 1 vs Phase 2 comparison:**

![Training Curves Comparison](plots/training_curves.png)

| Metric | Phase 1 (Roboflow) | Phase 2 (Client) | Δ |
|---|---|---|---|
| mAP@0.5 (epoch 50) | ~0.838 | ~0.857 | +0.019 |
| mAP@0.5-0.95 (epoch 50) | ~0.527 | ~0.549 | +0.022 |
| Precision (epoch 50) | ~0.91 | ~0.93 | +0.02 |
| Recall (epoch 50) | ~0.76 | ~0.79 | +0.03 |

---

## Results

### Final Evaluation

Full training dashboard showing all losses and metrics over 50 epochs:

![Results](plots/results.png)

#### F1-Confidence Curve

Optimal confidence threshold: **0.380** → F1 = **0.83**

![F1 Confidence Curve](plots/BoxF1_curve.png)

#### Precision-Confidence Curve

All classes reach precision **1.00** at confidence **0.938**.

![Precision Confidence Curve](plots/BoxP_curve.png)

#### Recall-Confidence Curve

All classes recall **0.86** at confidence **0.000**.

![Recall Confidence Curve](plots/BoxR_curve.png)

#### Precision-Recall Curve

![PR Curve](plots/BoxPR_curve.png)

| Class | AP@0.5 |
|---|---|
| Safety Vest | **0.935** |
| Mask | 0.914 |
| Person | 0.862 |
| Hardhat | 0.851 |
| NO-Safety Vest | 0.813 |
| NO-Mask | 0.758 |
| NO-Hardhat | 0.726 |
| **All classes** | **0.837** |

---

### Per-Class Metrics

Precision, Recall, and F1-Score per class at the optimal confidence threshold (0.380):

![Per Class Metrics](plots/per_class_metrics.png)

---

### Confusion Matrix

#### Raw Counts

![Confusion Matrix](plots/confusion_matrix.png)

#### Normalised (per true class = recall)

![Confusion Matrix Normalized](plots/confusion_matrix_normalized.png)

| Class | Recall | Primary Error |
|---|---|---|
| Mask | 0.90 | 1% → background |
| Safety Vest | 0.89 | 11% → background |
| Person | 0.81 | 17% → background (occlusion) |
| Hardhat | 0.78 | 13% → background, 3% → NO-Hardhat |
| NO-Mask | 0.76 | 18% → background |
| NO-Safety Vest | 0.71 | 17% → background |
| NO-Hardhat | **0.67** | **30% → background** (hardest class) |

> **Key insight:** The dominant failure mode is **background confusion** (missed detections), not class confusion. Objects are being missed rather than mis-labelled. This is best addressed by adding more training data and enabling mosaic/copy-paste augmentation.

---

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA 11.8+ (for GPU training)
- 4 GB VRAM minimum (GTX 1650 or equivalent)

### Setup

```bash
git clone https://github.com/your-org/ppe-detection.git
cd ppe-detection
pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
Pillow>=10.0.0
roboflow>=1.1.0
```

---

## Quick Start

### Run Inference on an Image

```python
from ultralytics import YOLO

model = YOLO("runs/phase2/weights/best.pt")

results = model.predict(
    source="image.jpg",
    conf=0.38,       # optimal F1 threshold
    iou=0.45,
    imgsz=640,
    save=True
)
results[0].show()
```

### Run on a Video Stream

```python
results = model.predict(
    source="rtsp://camera-ip:554/stream",  # or path to .mp4
    conf=0.38,
    stream=True
)

for r in results:
    frame = r.plot()   # annotated frame
    cv2.imshow("PPE Detection", frame)
```

### Train Phase 1

```bash
python train_phase1.py \
  --data data/roboflow/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

### Train Phase 2 (Fine-Tune)

```bash
python train_phase2.py \
  --data data/client/data.yaml \
  --weights runs/phase1/weights/best.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

---

## Inference & Deployment

### Recommended Confidence Thresholds

| Use Case | Confidence | IOU | Rationale |
|---|---|---|---|
| Real-time safety alert | **0.38** | 0.45 | Optimal F1 — balanced precision/recall |
| High-precision audit log | 0.60 | 0.50 | Fewer false alarms |
| High-recall compliance | 0.25 | 0.40 | Catch all violations, accept more FP |

### Export Formats

```python
from ultralytics import YOLO

model = YOLO("runs/phase2/weights/best.pt")

model.export(format="onnx")      # Cross-platform — recommended default
model.export(format="engine")    # TensorRT — 2-3× faster on NVIDIA hardware
model.export(format="tflite")    # TensorFlow Lite — mobile/edge
model.export(format="coreml")    # Apple devices
```

### Class Labels (`data.yaml`)

```yaml
names:
  0: Hardhat
  1: Mask
  2: NO-Hardhat
  3: NO-Mask
  4: NO-Safety Vest
  5: Person
  6: Safety Vest

nc: 7
```

---

## Repository Structure

```
ppe-detection/
├── data/
│   ├── roboflow/              # Phase 1 dataset (Roboflow export)
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   ├── client/                # Phase 2 client-specific images
│   │   ├── train/
│   │   ├── valid/
│   │   └── data.yaml
├── runs/
│   ├── phase1/
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── phase2/
│       └── weights/
│           ├── best.pt        ← use this for inference
│           └── last.pt
├── plots/                     # All diagnostic plots
│   ├── model_comparison.png
│   ├── training_curves.png
│   ├── results.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix.png
│   └── ...
├── train_phase1.py
├── train_phase2.py
├── predict.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

### Current Limitations

- **Background confusion** is the primary failure mode (15–30% of objects missed, not mis-classified)
- **NO-Hardhat recall = 0.67** — hardest class, needs more annotated examples
- **Mask underrepresented** (1,743 instances) — may limit generalisation across diverse mask styles
- **Validation loss gap** (~3.2 val vs ~2.1 train at epoch 50) suggests mild overfitting
- Single-GPU training on GTX 1650 limits batch size and speed

### Roadmap

- [ ] Collect 500–1,000 additional NO-Hardhat and Mask images
- [ ] Enable mosaic + copy-paste augmentation for better small-object detection
- [ ] Upgrade to YOLOv8s if deployment latency budget allows (>100 ms)
- [ ] Export to TensorRT engine for 2–3× speedup on NVIDIA edge devices
- [ ] Integrate ByteTrack / BoT-SORT for temporal consistency in video
- [ ] Active learning loop: feed deployment hard negatives back into training
- [ ] Add confidence calibration for reliable probability estimates

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>YOLOv8n · Two-Phase Training · GTX 1650 · mAP@0.5 0.837</sub>
</div>
