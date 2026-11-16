
# ✈️ AeroResilience AI
### Intelligent Airline Disruption Prediction, Recovery Optimization & Resilience Analytics

AeroResilience AI is an end-to-end airline operations research platform that combines **machine learning, network analytics, operations research, passenger recovery modeling, financial analysis, and interactive decision intelligence**.

The project demonstrates how airlines can use historical flight data and simulated disruption scenarios to evaluate recovery strategies, reduce cascading delays, assess passenger impacts, and understand financial trade-offs.

> **Research prototype:** Historical flight records are sourced from the U.S. Bureau of Transportation Statistics (BTS). Aircraft recovery scenarios, crew assignments, passenger demand, seat inventory, and financial parameters are simulated. The platform is not connected to live airline systems.

---

## Project Overview

Airline disruptions can propagate through aircraft rotations, crew schedules, passenger connections, and airport networks.

A localized disruption may therefore affect flights and passengers far beyond the original airport.

AeroResilience AI investigates six interconnected questions:

1. **Flight Disruption Intelligence:** Which flights and airport networks are most vulnerable to disruption?
2. **Aircraft & Crew Recovery:** How should available aircraft and crews be prioritized after an operational disruption?
3. **Passenger Rebooking:** How many simulated passengers can be rebooked onto alternative flights?
4. **Financial Impact:** What are the modeled cost differences between recovery policies?
5. **Resilience Simulation:** How do results change under different operational constraints and financial assumptions?
6. **Executive Command Center:** How can these findings be presented in an interactive decision-support dashboard?

---

## Project Architecture

```text
Historical BTS Flight Data
           |
           v
+-------------------------------+
| Module 1                      |
| Flight Disruption Intelligence|
| Prediction + Network Analysis |
+-------------------------------+
           |
           v
+-------------------------------+
| Module 2                      |
| Aircraft & Crew Recovery      |
| Optimization + Delay Cascades |
+-------------------------------+
           |
           v
+-------------------------------+
| Module 3                      |
| Passenger Rebooking           |
| Connection Risk + Allocation  |
+-------------------------------+
           |
           v
+-------------------------------+
| Module 4                      |
| Financial Recovery Analytics  |
| Cost Modeling + Sensitivity   |
+-------------------------------+
           |
           v
+-------------------------------+
| Module 5                      |
| Resilience & What-If Analysis |
| Scenario Comparisons          |
+-------------------------------+
           |
           v
+-------------------------------+
| Module 6                      |
| Executive Decision Dashboard  |
| Streamlit + Plotly            |
+-------------------------------+
```

---

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Optimization | Mathematical optimization and recovery heuristics |
| Network Modeling | NetworkX |
| Visualization | Plotly |
| Interactive Dashboard | Streamlit |
| Data Storage | CSV, Parquet |
| Development | VS Code, Git, GitHub |

---

## Module 1 — Flight Disruption Intelligence

This module analyzes historical BTS flight records and develops a disruption-risk prioritization workflow.

### Capabilities

- Historical flight disruption analysis
- Airport and route risk analysis
- Machine-learning-based disruption classification
- Model evaluation using ROC AUC and PR AUC
- Risk-based alert prioritization
- Network disruption visualization

### Dataset

The historical analysis covers approximately **1,847,936 U.S. flights** from June through August 2026.

### Model Performance

| Metric | Result |
|---|---:|
| ROC AUC | 0.6991 |
| PR AUC | 0.4554 |

These metrics describe performance on the project's evaluated historical dataset and should not be interpreted as live prediction accuracy.

---

## Module 2 — Aircraft & Crew Recovery Optimization

This module simulates how disruptions propagate through aircraft rotations and evaluates alternative recovery strategies.

### Recovery Policies

**First Come, First Served (FCFS)**  
Prioritizes flights based on their original order.

**Network Priority**  
Uses network-related impact scores to determine recovery priority.

**Fairness-Constrained Recovery**  
Introduces fairness considerations into flight prioritization.

**Aircraft-Aware Recovery**  
Considers aircraft rotations and downstream delay propagation when constructing recovery schedules.

### Example Ground-Stop Scenario

The primary recovery experiment simulates a hypothetical ground stop at Chicago O'Hare International Airport (ORD) on August 12, 2026.

The experiment considers **177 eligible departures**.

### Aircraft Recovery Results

For the 20-minute minimum-turnaround scenario:

| Metric | FCFS | Aircraft-Aware |
|---|---:|---:|
| Incremental network delay | 103,710 minutes | 96,092 minutes |
| Relative delay improvement | — | 7.35% |

The aircraft-aware policy reduced modeled incremental network delay by **7,618 minutes** in this simulated scenario.

This is a modeled reduction, not an observed airline operating result.

### Crew Recovery

The project also creates synthetic primary and reserve crew resources and tests the feasibility of recovery schedules under simplified availability and duty constraints.

Crew resilience experiments evaluate:

- Normal reserve availability
- Reduced reserve availability
- Late-arriving reserve crews
- Reduced reserve duty time

Crew assignments are synthetic and do not constitute an airline-certified crew legality system.

---

## Module 3 — Passenger Rebooking Intelligence

This module models plausible connecting passenger itineraries using historical flight schedules and synthetic passenger volumes.

### Capabilities

- Connecting-flight itinerary construction
- Missed-connection feasibility analysis
- Same-airline alternate flight identification
- Synthetic seat inventory allocation
- Greedy passenger rebooking
- Global passenger allocation optimization
- Connection-time sensitivity analysis

### Passenger Recovery Results

| Metric | FCFS | Aircraft-Aware |
|---|---:|---:|
| Synthetic passengers at risk | 3,383 | 3,379 |
| Passengers rebooked | 3,146 | 3,133 |
| Passengers unaccommodated | 237 | 246 |
| Rebooking rate | 92.99% | 92.72% |

### Key Finding

Aircraft-aware recovery reduced network delay but did not outperform FCFS on rebooking coverage in the tested scenario.

This highlights an operational trade-off:

**Improving aircraft delay performance does not automatically improve every passenger recovery metric.**

All passenger counts and seat inventories are simulated.

---

## Module 4 — Financial Impact & Recovery Analytics

This module converts simulated operational outcomes into illustrative cost estimates.

### Cost Components

- Operational delay costs
- Passenger rebooking expenses
- Passenger assistance expenses
- Reserve crew assignment costs

### Baseline Financial Results

| Recovery Policy | Modeled Total Cost |
|---|---:|
| FCFS | $4,978,780 |
| Aircraft-Aware | $4,637,180 |
| Modeled Difference | $341,600 |

The aircraft-aware strategy has a lower modeled total cost under the baseline cost assumptions.

### Financial Sensitivity

Across 72 hypothetical cost scenarios:

- Aircraft-aware had the lower modeled cost in 62 scenarios.
- FCFS had the lower modeled cost in 10 scenarios.

These values are sensitivity-analysis outputs, not verified airline expenditures or realized savings.

---

## Module 5 — Resilience & What-If Simulator

This module evaluates the sensitivity of modeled recovery performance to changing assumptions.

### Analyses

- Aircraft turnaround sensitivity
- Network delay propagation scenarios
- Passenger minimum connection time sensitivity
- Synthetic crew reserve availability
- Financial unit-cost scenarios
- Cross-policy comparisons

### Aircraft Turnaround Sensitivity

| Minimum Turnaround | Aircraft-Aware Delay Improvement |
|---|---:|
| 10 minutes | 7.12% |
| 20 minutes | 7.35% |
| 30 minutes | 7.27% |
| 40 minutes | 7.14% |

The results demonstrate consistent modeled delay improvement across the tested turnaround requirements.

### Important Limitation

The what-if analysis uses precomputed scenario outputs. Certain passenger and crew outcomes are held fixed instead of being recalculated for each financial assumption.

This is a scenario-comparison platform, not a fully coupled airline digital twin.

---

## Module 6 — Executive Decision Command Center

The Streamlit application brings the project results into an interactive dashboard.

### Dashboard Sections

1. Executive Overview
2. Flight Disruption Intelligence
3. Aircraft Recovery
4. Crew Recovery
5. Passenger Intelligence
6. Financial Analytics
7. What-If Simulator

### Dashboard Features

- Aviation-themed executive interface
- Interactive U.S. airport network map
- Recovery policy comparisons
- Aircraft turnaround controls
- Crew resilience scenarios
- Passenger rebooking analytics
- Financial cost breakdowns
- Financial resilience heatmap
- CSV export options

The airport network visualization uses saved historical flight records from the recovery scenario. It does not display live aircraft positions.

---

## Key Research Findings

**1. Recovery optimization is a multi-objective problem.**

The strategy that reduces aircraft delay may not maximize passenger rebooking coverage.

**2. Aircraft rotation constraints matter.**

Accounting for downstream aircraft dependencies can materially affect modeled network recovery.

**3. Crew feasibility can constrain recovery performance.**

Synthetic reserve crew availability and reporting delays influence which recovery schedules remain feasible.

**4. Financial preferences depend on assumptions.**

A recovery policy's modeled cost advantage varies across assumed cost rates.

**5. Integrated dashboards improve decision visibility.**

Combining model outcomes makes operational trade-offs easier to compare and communicate.

---

## Data Source

### U.S. Bureau of Transportation Statistics

Historical U.S. airline on-time performance records form the primary data source.

Official source:

https://www.transtats.bts.gov/

The project uses June, July, and August 2026 historical flight data.

### Data Availability

Large raw and processed datasets are excluded from this repository using `.gitignore`.

Some workflows require the original BTS datasets and previously generated intermediate outputs.

The repository is therefore provided as a research codebase rather than a fully self-contained, one-command reproduction package.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/aeroresilience-ai.git
cd aeroresilience-ai
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Run the Dashboard

When the required analytical reports are available locally:

```bash
python -m streamlit run app.py
```

The application will normally open at:

http://localhost:8501

Because generated outputs are not included in the default Git repository, certain charts and metrics will show unavailable-data messages until the corresponding reports have been generated or restored.

---

## Run Final Project Validation

```bash
python -m src.validate_final_project
```

The final local validation run completed with:

| Metric | Result |
|---|---:|
| Checks executed | 54 |
| Checks passed | 54 |
| Checks failed | 0 |

The checks cover available files, schemas, mathematical reconciliation, dashboard syntax, and related consistency conditions.

Passing these checks does not establish production readiness or real-world predictive or recovery performance.

---

## Research Limitations

AeroResilience AI is an educational and analytical research prototype.

The project does not include:

- Real-time airline scheduling or dispatch feeds
- Actual passenger booking records
- Actual airline seat availability
- Certified crew duty and legality rules
- Complete airport gate and airspace constraints
- Verified airline operational cost data
- A fully coupled live recovery optimizer
- Production monitoring or enterprise deployment guarantees

Results should be interpreted as controlled simulation and modeling outcomes.

---

## Potential Future Extensions

- Real-time flight and weather data integration
- More detailed aircraft and crew feasibility constraints
- Passenger-specific connection and rebooking priorities
- Multi-airline and codeshare rebooking
- Probabilistic disruption scenario generation
- Multi-objective recovery optimization
- Continuous simulation calibration
- Production-grade deployment and monitoring

---

## Project Status

**Six analytical modules implemented.**

**Executive dashboard developed and visually reviewed.**

**Final local validation: 54 checks passed, 0 failed.**

The project is ready for repository packaging and research demonstration, subject to the availability of its locally generated analytical reports.

---

## Author

**Devarshi Trivedi**

Business Analytics | Machine Learning | Operations Research | Decision Intelligence

---

*AeroResilience AI demonstrates how historical aviation data, simulation, optimization and analytics can support the evaluation of airline recovery decisions.*
