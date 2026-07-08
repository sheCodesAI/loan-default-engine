Read and follow `idbi_plann.md` as the project's single source of truth.

The AI module has already been generated, dependencies fixed, `loan_data.csv` integrated, the model trained, inference verified, and all tests passed. Do NOT restart development or retrain the model unless absolutely necessary.

Your task is to perform a final production-quality engineering pass on the entire AI module.

Tasks:

1. Verify the entire project against `idbi_plann.md` and ensure every planned module has been implemented correctly.

2. Review and improve the overall architecture, modularity, scalability, maintainability, code quality, performance, documentation, logging, type hints, comments, and error handling.

3. Detect and remove duplicate logic, dead code, unnecessary files, and improve the project wherever beneficial without breaking the architecture.

4. Verify the complete AI workflow:
   - Borrower 360° Intelligence
   - Geo & Resilience Intelligence
   - Business Rules Engine
   - Recommendation Engine
   - AI Risk Intelligence
   - SHAP Explainability
   - What-If Simulator
   - Training Pipeline
   - Inference Pipeline

5. Ensure all generated artifacts exist and are valid:
   - Trained Model
   - Preprocessor
   - Metadata
   - Metrics
   - SHAP Outputs
   - EDA Outputs

6. Create a professional `demo.py` that demonstrates the complete AI system end-to-end using the trained model.

The demo should:
- Load the trained model automatically.
- Load one sample borrower (or one row from `loan_data.csv`).
- Run the complete inference pipeline.
- Display:
  • Borrower Score
  • Estimated Cash Flow Health
  • Financial Health
  • Repayment Capacity
  • Trust Score
  • Geo & Resilience Score
  • Default Probability
  • Risk Level
  • Expected Loss
  • SHAP Top Risk Factors
  • Business Rule Results
  • Final Recommendation
- Run one What-If simulation (e.g., reduce loan amount or add co-applicant) and clearly show before vs after results.

The demo should be clean, readable, and suitable for live hackathon presentation.

7. Create a `walkthrough.md` that explains:
- Project architecture
- Folder structure
- AI workflow
- Intelligence engines
- Feature engineering
- Model training
- Explainability
- Business rules
- Recommendation logic
- What-If simulator
- How to run the project
- How to demonstrate it during judging

8. Create a `PROJECT_HEALTH_REPORT.md` containing:
- Overall architecture review
- Model performance
- Evaluation metrics
- Completed modules
- Remaining limitations
- Future enhancements
- Production readiness score
- Hackathon readiness score
- Suggestions for future improvements

9. Verify that the project can be executed from scratch using only:
- pip install -r requirements.txt
- python -m ai.pipeline.train_pipeline
- python demo.py

10. Perform one final engineering review and make any improvements that would be expected from a senior AI engineer building a production-grade banking AI system.

Do not recreate existing modules or duplicate code. Reuse and improve the current implementation wherever possible.

Only stop after the AI module is polished, fully functional, well documented, demo-ready, and hackathon-ready.
