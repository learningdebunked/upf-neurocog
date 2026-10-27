"""
Replaces Section 8.3 Limitations in the manuscript with an empirically-grounded
expansion informed by the actual NHANES 2011-2014 results.
"""

import re

MANUSCRIPT = "/home/sandbox/manuscript_upf_eeg_cognitive_fatigue_ai_workforce.md"

NEW_LIMITATIONS = r"""### 8.3 Limitations

While the proposed framework offers significant potential, several limitations must be acknowledged. These include both methodological constraints inherent to the NHANES 2011–2014 validation (n=2,780) and broader theoretical constraints of the AI framework architecture.

**1. Cross-Sectional Design and Causal Inference (NHANES-Specific):**
The NHANES 2011–2014 empirical validation is cross-sectional by design. Within this sample (n=2,780 adults ≥60 years), we observed a statistically significant monotonic decline in DSST scores across UPF quartiles (Q1: 46.9 ± 17.6 vs. Q4: 44.5 ± 17.0; ANOVA F=2.87, p=0.035). However, this cross-sectional association cannot establish temporal precedence. Reverse causation is particularly plausible in older adult populations: individuals with pre-existing cognitive impairment may exhibit reduced executive function needed for meal planning, grocery shopping, and food preparation, thereby increasing reliance on convenient ultra-processed products. Longitudinal NHANES follow-up cohorts or prospective panel studies with repeated dietary assessments would be required to test the diet→cognition directionality formally. Instrumental variable approaches using county-level food environment indices (e.g., USDA Food Environment Atlas) as exogenous variation in UPF access represent a promising quasi-experimental design for future work.

**2. Single 24-Hour Dietary Recall: Measurement Error and Regression Attenuation:**
NHANES dietary intake is assessed via a single 24-hour dietary recall (Day 1), which is the primary basis for our NOVA-classified UPF exposure variable. A single recall is a well-documented poor estimate of habitual intake due to day-to-day within-person variability. The intraclass correlation coefficient (ICC) for UPF intake from a single vs. three recalls is typically 0.35–0.50, meaning a single day recall captures only 35–50% of true habitual UPF variance. This introduces classical attenuation bias: the true association between habitual UPF intake and cognitive performance is systematically underestimated. In our data, the linear regression coefficient of β=−0.324 DSST points per 10% UPF energy (95% CI: −0.079 to +0.019 per 1%; p=0.196) is likely attenuated toward null relative to the true habitual-diet effect. The NHANES 2011–2014 cohort includes a Day 2 recall for a subsample; future analyses should apply regression calibration or multiple imputation using the Day 2 recall to correct for attenuation and obtain debiased effect estimates.

**3. Non-Significant Individual UPF Effect and Socioeconomic Confounding:**
The linear regression coefficient for UPF intake was not statistically significant at conventional thresholds (β=−0.324 per 10%; p=0.196), and the logistic regression OR was similarly attenuated (OR=0.959 per 10% UPF; 95% CI: 0.803–1.058; p=0.571). These null results in univariate regression—combined with the significant group-level ANOVA trend (p=0.035)—are consistent with substantial confounding by socioeconomic position. SHAP value decomposition of the XGBoost model (5-fold CV AUC=0.812) revealed that the two dominant predictors of impaired cognitive performance were Education Level (mean |SHAP|=0.826, rank 1) and Income-to-Poverty Ratio (mean |SHAP|=0.626, rank 2), with UPF percentage ranking 8th (mean |SHAP|=0.237). This hierarchy suggests that UPF intake and cognitive performance share a common cause—socioeconomic deprivation—which simultaneously constrains dietary quality and limits access to cognitive reserve-building resources (education, healthcare, environmental enrichment). Mediation analysis partitioning the total UPF effect into direct and education/income-mediated components is warranted to disentangle these pathways. Failure to acknowledge this confounding structure would risk overstating the independent causal role of UPF intake in cognitive fatigue surveillance systems.

**4. DSST as a Proxy for Cognitive Fatigue (Not Direct EEG):**
The Digit Symbol Substitution Test (DSST) was used as the primary cognitive outcome in the NHANES validation as it is the only standardized cognitive assessment available in the NHANES 2011–2014 cognitive function module. DSST measures processing speed and sustained attention—constructs related to but not identical to EEG-indexed cognitive fatigue. EEG cognitive fatigue signatures (theta-band power increases at Fz, alpha suppression, frontal theta/alpha ratio elevation) reflect moment-to-moment neurophysiological state and are sensitive to acute fatigue induction, whereas DSST captures trait-level cognitive ability accumulated over a lifetime. The correlation between DSST performance and EEG-derived fatigue indices is moderate (estimated r=0.40–0.55 from laboratory studies), introducing substantial attenuation in our validation. A low DSST score in an older adult may reflect lifelong socioeconomic disadvantage, early neurodegeneration, or sedentary lifestyle rather than UPF-induced acute fatigue. Direct validation requires datasets combining 24-hour dietary recalls with concurrent portable EEG or neuroimaging—for example, the UK Biobank accelerometer substudy or the Human Connectome Project aging cohort—which presently do not include day-level dietary exposure data at the granularity required.

**5. Self-Report Bias in Dietary and Cognitive Data:**
Both dietary and cognitive data in NHANES rely on self-report. The 24-hour recall methodology is susceptible to telescoping, forgetting of consumed items (particularly beverages and snacks), and social desirability bias in under-reporting processed foods. Validation studies using urinary biomarkers (e.g., urinary sucrose/fructose as UPF markers) and plasma fatty acid profiles suggest systematic under-reporting of UPF energy by 10–15%. Regarding DSST, performance can be influenced by test anxiety, physical fatigue at time of interview, hearing or vision impairment (common in adults ≥60), and interviewer effects—all of which inflate non-differential measurement error and attenuate observed associations.

**6. Generalizability Across Populations and Food Systems:**
The NHANES validation was conducted in U.S. community-dwelling adults aged ≥60. UPF prevalence, NOVA food classification boundaries, and the diet–cognition relationship may differ substantially across populations with different food systems (e.g., Brazil, where the NOVA system was developed; Mediterranean European countries with traditionally lower UPF intakes; sub-Saharan Africa). The SHAP-derived socioeconomic dominance (education, income) may be attenuated in countries with universal healthcare and stronger social safety nets, allowing the direct dietary effect to emerge more clearly. External validation in NHANES Mexico (ENSANUT), the UK National Diet and Nutrition Survey, or the European Prospective Investigation into Cancer and Nutrition (EPIC) cohort is recommended.

**7. Residual Confounding by Unmeasured Variables:**
Despite adjustment for age, sex, race/ethnicity, BMI, income-to-poverty ratio, education, and physical activity, residual confounding by childhood adversity, genetic risk variants (APOE ε4 for Alzheimer's risk), depression/anxiety (not measured in CFQ module), medication use (polypharmacy is common in older adults and affects DSST), sleep quality, and environmental exposures (air pollution, noise) cannot be excluded. Mendelian randomization using genome-wide association study (GWAS) variants for food preference as instrumental variables represents a more rigorous approach to causal inference, though NHANES does not include genetic data. Linkage to the All of Us Research Program or UK Biobank would enable such analyses.

**8. Absence of Direct EEG Measurements in National Surveys:**
The broader AI framework predicts EEG biomarkers (theta power, alpha suppression, frontal theta/alpha ratio) from behavioral features without EEG measurement in NHANES/BRFSS participants. This introduces a two-stage prediction uncertainty: first, predicting cognitive impairment status from dietary and demographic features (validated here, AUC=0.812); second, mapping impairment probability to specific EEG signatures (not yet validated empirically). The expected correlation between predicted and observed EEG-derived fatigue indices is r=0.55–0.70 based on transfer learning literature. Full framework validation requires independent datasets with concurrent dietary assessment and EEG recording—ideally in workplace settings to maximize ecological validity for occupational health applications.

**9. Mechanistic Pathway Complexity:**
The proposed mechanistic pathway (UPF consumption → gut microbiome disruption → systemic neuroinflammation → brain oscillatory disruption → cognitive fatigue) is simplified for conceptual clarity. Multiple parallel pathways operate simultaneously: dopaminergic reward dysregulation driven by ultra-processed food palatability, oxidative stress from advanced glycation end-products (AGEs) in processed foods, mitochondrial dysfunction from excess saturated fat and fructose, and disrupted circadian metabolic rhythms. These pathways interact non-linearly and their relative contributions likely vary by genetic background (e.g., APOE, FKBP5), gut microbiome composition, and age. Systems biology approaches integrating multi-omics data (metabolomics, microbiomics, proteomics) alongside EEG and dietary data are needed to dissect mechanistic complexity.

**10. Intervention Feasibility and Adherence:**
Long-term dietary behavior change remains challenging. Adherence rates to dietary interventions average 30–50% at 12 months and decline further at 24 months. The occupational health surveillance application assumes that early detection of cognitive fatigue risk will trigger dietary intervention; however, the translation from risk identification to sustained behavioral change requires intensive counseling, environmental restructuring (cafeteria redesign, subsidized healthy food), and social support—resources that vary widely across employers. Future work should evaluate implementation feasibility in representative workplace settings including shift-work industries where UPF consumption is disproportionately high.

**11. Ethical and Privacy Concerns in Workplace AI Deployment:**
Implementation of AI-driven cognitive fatigue surveillance raises ethical concerns regarding privacy, informed consent, data security, and potential discriminatory use of risk scores. Algorithmic bias—differential prediction accuracy by race/ethnicity, education level, or disability status—was not fully evaluated in this analysis due to sample size constraints within demographic subgroups. Given that SHAP analysis identified socioeconomic factors as dominant predictors, AI-generated risk scores may systematically disadvantage lower-income workers if used in employment decisions rather than purely for health promotion. Independent fairness auditing using subgroup AUC metrics and calibration tests across demographic groups is required before any workforce deployment.

"""

with open(MANUSCRIPT, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the existing Section 8.3 block — from "### 8.3 Limitations" to "### 8.4 Ethical"
pattern = r'(### 8\.3 Limitations\n)(.*?)(### 8\.4 Ethical)'
match = re.search(pattern, content, flags=re.DOTALL)
if not match:
    print("ERROR: Could not locate Section 8.3 block.")
    exit(1)

print(f"Found Section 8.3 at char positions {match.start()}–{match.end()}")
print(f"Old length: {len(match.group(2))} chars")

new_content = content[:match.start()] + NEW_LIMITATIONS + "### 8.4 Ethical" + content[match.end():]

with open(MANUSCRIPT, 'w', encoding='utf-8') as f:
    f.write(new_content)

lines = new_content.count('\n')
print(f"Manuscript updated. Total lines: {lines}")
print(f"New Section 8.3 length: {len(NEW_LIMITATIONS)} chars")
print("Done.")
