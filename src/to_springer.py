"""
to_springer.py -- Converts MANUSCRIPT_v2_final.md into Springer Nature house format.

Changes applied:
  * Tables renumbered sequentially by order of appearance (Springer requires
    Table 1..N; the generated labels are 1, 2, 3a, 3b, ... A).
  * Table captions restyled to Springer form ("Table 1 Caption", caption above
    the table, no terminal period on the number).
  * "Figure 1" -> "Fig. 1".
  * Abstract trimmed to the ~250-word limit most Springer Nature journals apply.
  * References converted to Springer basic numbered style
    (Author AB, Author CD (Year) Title. Journal Vol:pages).
  * Front matter and a Declarations section in Springer order.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "MANUSCRIPT_v2_final.md"
RES_DIR = ROOT / "results"
OUT = ROOT / "paper" / "MANUSCRIPT_springer.md"

# Order of appearance -> Springer table number
# Reliability tables moved to the supplement, so the calibration table that
# followed them becomes Table 14.
# Order of appearance -> Springer table number
TABLE_MAP = {"A": 1, "1": 2, "2": 3, "2b": 4, "2c": 5,
             "3a": 6, "3b": 7, "3c": 8, "4a": 9, "4b": 10,
             "5a": 11, "5b": 12, "6": 13, "7": 14, "8": 15, "10": 16}

ABSTRACT = """\
Deploying clinical screening models on microcontrollers requires networks small
enough for tens of kilobytes of Flash and integer-only arithmetic. We present
TinyMed-Net, a 3,942-parameter depthwise-separable network for tabular clinical
data, evaluated on the PIMA Indians Diabetes (n=768) and UCI Chronic Kidney
Disease (n=400) cohorts under ten repeats of stratified five-fold
cross-validation with paired significance testing. We report four results that
qualify the conventional account of tiny-model deployment. First, TinyMed-Net is
statistically indistinguishable from logistic regression on PIMA (ΔF1 = +0.006,
p = 0.63) and significantly worse on chronic kidney disease (ΔF1 = −0.097,
p = 0.002): at this cohort size the network buys no accuracy over a linear model,
only a fixed, quantizable inference graph. Second, we find no evidence that
quantization-aware training is necessary at this scale. Under a validated
toolchain with batch-normalisation folding and per-channel symmetric weights,
post-training quantization matches FP32 across 20 seeds (PIMA AUROC 0.804 versus
0.803), and we show that a reported post-training-quantization collapse can be an
artifact of quantizer implementation rather than a property of the method. Third,
parameter-count footprint estimates are optimistic: our exported INT8 TensorFlow
Lite model holds 4.19 KiB of parameters but occupies 19.59 KiB, because graph
structure and quantization metadata dominate at this scale. Fourth, we report
substantial age-related disparity on PIMA (equalized-odds difference 0.271, 95%
CI [0.201, 0.358]) and severe miscalibration on chronic kidney disease (expected
calibration error 0.297), the latter fully corrected by cross-fitted temperature
scaling (0.009). We additionally document that a diabetes dataset in wide use in
this literature is synthetic."""

REFERENCES = """\
1. Wilson ML, Fleming KA, Kuti MA, Looi LM, Lago N, Ru K (2018) Access to
   pathology and laboratory medicine services: a crucial gap. Lancet
   391(10133):1927–1938. https://doi.org/10.1016/S0140-6736(18)30458-6
2. World Health Organization (2021) The selection and use of essential in vitro
   diagnostics. WHO Technical Report Series, Geneva
3. Warden P, Situnayake D (2019) TinyML: machine learning with TensorFlow Lite
   on Arduino and ultra-low-power microcontrollers. O'Reilly Media, Sebastopol
4. Banbury C, Reddi VJ, Torelli P, et al (2021) MLPerf Tiny benchmark. In:
   Proceedings of the Neural Information Processing Systems track on datasets
   and benchmarks, vol 1. arXiv:2106.07597
5. David R, Duke J, Jain A, et al (2021) TensorFlow Lite Micro: embedded machine
   learning on TinyML systems. In: Proceedings of machine learning and systems
   (MLSys), vol 3. arXiv:2010.08678
6. Lai L, Suda N, Chandra V (2018) CMSIS-NN: efficient neural network kernels for
   Arm Cortex-M CPUs. arXiv:1801.06601
7. Lin J, Chen WM, Lin Y, Cohn J, Gan C, Han S (2020) MCUNet: tiny deep learning
   on IoT devices. Advances in neural information processing systems
   33:11711–11722
8. Banbury C, Zhou C, Fedorov I, et al (2021) MicroNets: neural network
   architectures for deploying TinyML applications on commodity
   microcontrollers. Proceedings of machine learning and systems (MLSys)
   3:517–532
9. Howard AG, Zhu M, Chen B, et al (2017) MobileNets: efficient convolutional
   neural networks for mobile vision applications. arXiv:1704.04861
10. Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC (2018) MobileNetV2: inverted
    residuals and linear bottlenecks. In: Proceedings of the IEEE/CVF conference
    on computer vision and pattern recognition, pp 4510–4520
11. Hu J, Shen L, Sun G (2018) Squeeze-and-excitation networks. In: Proceedings
    of the IEEE/CVF conference on computer vision and pattern recognition,
    pp 7132–7141
12. He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image
    recognition. In: Proceedings of the IEEE conference on computer vision and
    pattern recognition, pp 770–778
13. Ioffe S, Szegedy C (2015) Batch normalization: accelerating deep network
    training by reducing internal covariate shift. In: Proceedings of the
    international conference on machine learning, pp 448–456
14. Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural
    network. arXiv:1503.02531
15. Jacob B, Kligys S, Chen B, et al (2018) Quantization and training of neural
    networks for efficient integer-arithmetic-only inference. In: Proceedings of
    the IEEE/CVF conference on computer vision and pattern recognition,
    pp 2704–2713. arXiv:1712.05877
16. Krishnamoorthi R (2018) Quantizing deep convolutional networks for efficient
    inference: a whitepaper. arXiv:1806.08342
17. Nagel M, Fournarakis M, Amjad RA, Bondarenko Y, van Baalen M, Blankevoort T
    (2021) A white paper on neural network quantization. arXiv:2106.08295
18. Gholami A, Kim S, Dong Z, Yao Z, Mahoney MW, Keutzer K (2021) A survey of
    quantization methods for efficient neural network inference.
    arXiv:2103.13630
19. Smith JW, Everhart JE, Dickson WC, Knowler WC, Johannes RS (1988) Using the
    ADAP learning algorithm to forecast the onset of diabetes mellitus. In:
    Proceedings of the annual symposium on computer application in medical care,
    pp 261–265
20. Rubini L, Soundarapandian P, Eswaran P (2015) Chronic kidney disease
    [dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5G020
21. Kelly M, Longjohn R, Nottingham K. The UCI machine learning repository.
    University of California, Irvine. https://archive.ics.uci.edu
22. American Diabetes Association Professional Practice Committee. Classification
    and diagnosis of diabetes: standards of care in diabetes. Diabetes Care
23. Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group (2013)
    KDIGO clinical practice guideline for the evaluation and management of
    chronic kidney disease. Kidney Int Suppl 3:1–150
24. Levey AS, Stevens LA, Schmid CH, et al (2009) A new equation to estimate
    glomerular filtration rate. Ann Intern Med 150(9):604–612.
    https://doi.org/10.7326/0003-4819-150-9-200905050-00006
25. Thomas MC, Brownlee M, Susztak K, et al (2015) Diabetic kidney disease. Nat
    Rev Dis Primers 1:15018. https://doi.org/10.1038/nrdp.2015.18
26. Breiman L (2001) Random forests. Mach Learn 45(1):5–32.
    https://doi.org/10.1023/A:1010933404324
27. Chen T, Guestrin C (2016) XGBoost: a scalable tree boosting system. In:
    Proceedings of the ACM SIGKDD international conference on knowledge discovery
    and data mining, pp 785–794
28. Grinsztajn L, Oyallon E, Varoquaux G (2022) Why do tree-based models still
    outperform deep learning on typical tabular data? In: Advances in neural
    information processing systems, datasets and benchmarks track
29. Shwartz-Ziv R, Armon A (2022) Tabular data: deep learning is not all you
    need. Inf Fusion 81:84–90. https://doi.org/10.1016/j.inffus.2021.11.011
30. Geirhos R, Jacobsen JH, Michaelis C, et al (2020) Shortcut learning in deep
    neural networks. Nat Mach Intell 2(11):665–673.
    https://doi.org/10.1038/s42256-020-00257-z
31. Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK (2018) Variable
    generalization performance of a deep learning model to detect pneumonia in
    chest radiographs: a cross-sectional study. PLoS Med 15(11):e1002683.
    https://doi.org/10.1371/journal.pmed.1002683
32. DeGrave AJ, Janizek JD, Lee SI (2021) AI for radiographic COVID-19 detection
    selects shortcuts over signal. Nat Mach Intell 3:610–619.
    https://doi.org/10.1038/s42256-021-00338-7
33. Roberts M, Driggs D, Thorpe M, et al (2021) Common pitfalls and
    recommendations for using machine learning to detect and prognosticate for
    COVID-19 using chest radiographs and CT scans. Nat Mach Intell
    3(3):199–217
34. Kapoor S, Narayanan A (2023) Leakage and the reproducibility crisis in
    machine-learning-based science. Patterns 4(9):100804.
    https://doi.org/10.1016/j.patter.2023.100804
35. Guo C, Pleiss G, Sun Y, Weinberger KQ (2017) On calibration of modern neural
    networks. In: Proceedings of the international conference on machine learning,
    pp 1321–1330
36. Niculescu-Mizil A, Caruana R (2005) Predicting good probabilities with
    supervised learning. In: Proceedings of the international conference on
    machine learning, pp 625–632
37. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW (2019)
    Calibration: the Achilles heel of predictive analytics. BMC Med 17(1):230.
    https://doi.org/10.1186/s12916-019-1466-7
38. Naeini MP, Cooper GF, Hauskrecht M (2015) Obtaining well calibrated
    probabilities using Bayesian binning. In: Proceedings of the AAAI conference
    on artificial intelligence, vol 29
39. Brier GW (1950) Verification of forecasts expressed in terms of probability.
    Mon Weather Rev 78(1):1–3
40. Hardt M, Price E, Srebro N (2016) Equality of opportunity in supervised
    learning. In: Advances in neural information processing systems 29,
    pp 3315–3323
41. Barocas S, Hardt M, Narayanan A (2023) Fairness and machine learning:
    limitations and opportunities. MIT Press, Cambridge
42. Chouldechova A (2017) Fair prediction with disparate impact: a study of bias
    in recidivism prediction instruments. Big Data 5:153–163
43. Obermeyer Z, Powers B, Vogeli C, Mullainathan S (2019) Dissecting racial bias
    in an algorithm used to manage the health of populations. Science
    366(6464):447–453. https://doi.org/10.1126/science.aax2342
44. Matthews BW (1975) Comparison of the predicted and observed secondary
    structure of T4 phage lysozyme. Biochim Biophys Acta 405(2):442–451
45. Chicco D, Jurman G (2020) The advantages of the Matthews correlation
    coefficient (MCC) over F1 score and accuracy in binary classification
    evaluation. BMC Genomics 21(1):6.
    https://doi.org/10.1186/s12864-019-6413-7
46. Wilcoxon F (1945) Individual comparisons by ranking methods. Biom Bull
    1:80–83
47. Demšar J (2006) Statistical comparisons of classifiers over multiple data
    sets. J Mach Learn Res 7:1–30
48. Efron B, Tibshirani RJ (1993) An introduction to the bootstrap. Chapman and
    Hall/CRC, Boca Raton
49. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP (2002) SMOTE: synthetic minority
    over-sampling technique. J Artif Intell Res 16:321–357
50. Paszke A, Gross S, Massa F, et al (2019) PyTorch: an imperative style,
    high-performance deep learning library. In: Advances in neural information
    processing systems
51. Pedregosa F, Varoquaux G, Gramfort A, et al (2011) Scikit-learn: machine
    learning in Python. J Mach Learn Res 12:2825–2830
52. Abadi M, Barham P, Chen J, et al (2016) TensorFlow: a system for large-scale
    machine learning. In: Proceedings of the USENIX symposium on operating
    systems design and implementation, pp 265–283
53. Collins GS, Reitsma JB, Altman DG, Moons KGM (2015) Transparent reporting of
    a multivariable prediction model for individual prognosis or diagnosis
    (TRIPOD): the TRIPOD statement. Ann Intern Med 162(1):55–63
54. Vandewiele G, Dehaene I, Kovács G, et al (2021) Overly optimistic prediction
    results on imbalanced data: a case study of flaws and benefits when applying
    over-sampling. Artif Intell Med 111:101987.
    https://doi.org/10.1016/j.artmed.2020.101987
55. Varoquaux G, Cheplygina V (2022) Machine learning for medical imaging:
    methodological failures and recommendations for the future. NPJ Digit Med
    5(1):48. https://doi.org/10.1038/s41746-022-00592-y"""

DECLARATIONS = """\
## Declarations

**Funding.** This research received no specific grant from any funding agency in
the public, commercial, or not-for-profit sectors. *(Amend if any author was
grant-supported.)*

**Competing interests.** The authors declare that they have no competing
interests.

**Use of generative AI.** As stated in Sect. 4.2, Claude (Anthropic) was used to
implement the analysis pipeline, execute the reported experiments, and prepare an
initial draft of this manuscript. All authors independently reproduced the
reported results and take full responsibility for the content. No AI system is
an author.

**Ethics approval.** This study analysed three publicly available, fully
de-identified datasets containing no personally identifiable information. No
human participants were recruited and no new data were collected, so
institutional review board approval was not required. Authors should confirm
whether their institution requires a formal exemption determination for
secondary analysis of public data.

**Consent to participate.** Not applicable; no human participants were recruited.

**Consent for publication.** Not applicable.

**Data availability.** All three datasets are publicly available. The PIMA
Indians Diabetes and UCI Chronic Kidney Disease datasets are distributed through
the UCI Machine Learning Repository [20, 21]. The harmonised cohorts used here
are reproduced in the code repository below.

**Code availability.** All code, frozen prediction artifacts, and the scripts
that regenerate every table in this article are available at
https://github.com/kapopat1980/tinymednet-2disease, which should be tagged at the
submitted revision.

**Author contributions.** *Draft for the authors to confirm and amend before
submission; a contributions statement is a declaration of fact and must reflect
what each author actually did.* Following CRediT: **Vivek Gondalia** —
Conceptualization, Methodology, Software, Formal analysis, Investigation,
Visualization, Writing – original draft. **Kalpesh Popat** — Conceptualization,
Methodology, Validation, Data curation, Writing – review and editing,
Supervision, Project administration. **Ashwin Dobariya** — Methodology,
Validation, Formal analysis, Writing – review and editing, Supervision. All
authors read and approved the final manuscript."""

FRONT = """\
# TinyMed-Net: What a Sub-4K-Parameter Network Can and Cannot Do for Tabular Clinical Screening

**Vivek Gondalia**, **Kalpesh Popat**, **Ashwin Dobariya**

*Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India*

**Corresponding author:** Kalpesh Popat, Faculty of Computer Applications,
Marwadi University, Rajkot, Gujarat, India. Email: __________. ORCID: __________

> Remaining fields to complete before submission: the corresponding author's
> email and each author's ORCID. Confirm the author order and the contributions
> statement below.

## Abstract

{abstract}

**Keywords** TinyML · Quantization · Post-training quantization · Model
compression · Point-of-care diagnostics · Microcontroller inference ·
Calibration · Algorithmic fairness

"""


def main():
    t = SRC.read_text(encoding="utf-8")

    # Body starts after the generated abstract/keywords block.
    body = t.split("## 1. Introduction", 1)[1]
    body = "## 1 Introduction" + body
    body = body.split("## References")[0]

    # Strip the old back matter; Springer collects it under Declarations.
    for h in ["## Data and code availability", "## Ethics statement", "## Funding",
              "## Declaration of competing interests", "## Author contributions"]:
        body = body.split(h)[0] if h == "## Data and code availability" else body

    # Table renumbering. Two passes via sentinels so no mapping collides.
    def sentinel(m):
        lbl = m.group(1)
        return f"@@TBL{TABLE_MAP[lbl]}@@" if lbl in TABLE_MAP else m.group(0)

    # Captions become bold paragraphs, not headings: a table caption in a
    # heading slot would appear in the table of contents and break section numbering.
    body = re.sub(r"## Table ([0-9]+[a-z]?)\.", sentinel, body)
    body = body.replace("**Table A.**", "@@TBL1@@")
    body = re.sub(r"@@TBL(\d+)@@", r"**Table \1**", body)

    # In-text cross-references
    body = body.replace("F1 in Tables 5a–b", "F1 in Tables 11 and 12")
    body = body.replace("**Figure 1.**", "**Fig. 1**")
    body = body.replace("![TinyMed-Net architecture](artifacts/figure1_architecture.png)",
                        "![](artifacts/figure1_architecture.png)")

    # Springer numbers sections without a period after the digit.
    body = re.sub(r"^(#{2,3}) (\d)\.(\d?)", lambda m: f"{m.group(1)} {m.group(2)}"
                  + (f".{m.group(3)}" if m.group(3) else ""), body, flags=re.M)
    body = re.sub(r"\bSection(\s+)", r"Sect.\1", body)

    # Drop markdown horizontal rules: pandoc renders them as black bars, and
    # numbered headings already delimit sections.
    body = re.sub(r"^-{3,}$\n?", "", body, flags=re.M)

    # Appendix: the supplementary tables, appended so the submission is one file.
    supp = RES_DIR / "supplementary.md"
    appendix = ""
    if supp.exists():
        sup = supp.read_text(encoding="utf-8")
        sup = sup.split("\n", 1)[1] if sup.startswith("# ") else sup
        sup = re.sub(r"^_Generated by.*?$", "", sup, flags=re.M)
        # "## Table S1. Caption" -> bold paragraph, matching the main-text style
        sup = re.sub(r"## Table (S\d+)\.", r"**Table \1**", sup)
        appendix = ("\n\n## Appendix A. Supplementary tables\n\n"
                    "_These tables report the complete metric set behind Tables 6 to 8 "
                    "and the binned reliability values behind Fig. 2. They are "
                    "generated by the same script from the same frozen predictions._\n"
                    + sup.strip() + "\n")

    out = (FRONT.format(abstract=ABSTRACT) + "\n" + body.rstrip() + appendix
           + "\n\n" + DECLARATIONS + "\n\n## References\n\n"
           + REFERENCES + "\n")
    OUT.write_text(out, encoding="utf-8")

    n_tables = len(re.findall(r"\*\*Table \d+\*\*", out))
    print(f"written {OUT.name}: {n_tables} tables, "
          f"{len(ABSTRACT.split())} words in abstract")


if __name__ == "__main__":
    main()
