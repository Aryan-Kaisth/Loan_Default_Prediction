from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal

class LoanRequest(BaseModel):
    """
    Loan request model with raw input features and computed financial metrics.
    """

    # --- Original Input Fields ---
    Age: Annotated[int, Field(..., ge=18, description="Age of applicant in years")]
    Income: Annotated[float, Field(..., ge=0, description="Annual income of applicant")]
    LoanAmount: Annotated[float, Field(..., ge=0, description="Requested loan amount")]
    CreditScore: Annotated[int, Field(..., ge=0, le=1000, description="Credit score of applicant")]
    MonthsEmployed: Annotated[int, Field(..., ge=0, description="Total months employed")]
    NumCreditLines: Annotated[int, Field(..., ge=0, description="Number of open credit lines")]
    InterestRate: Annotated[float, Field(..., ge=0, description="Loan interest rate in percentage")]
    LoanTerm: Annotated[int, Field(..., ge=1, description="Loan term in months")]
    DTIRatio: Annotated[float, Field(..., ge=0, description="Debt-to-Income ratio")]
    Education: Annotated[
        Literal["High School", "Bachelor's", "Master's", "PhD"],
        Field(..., description="Highest education level"),
    ]
    EmploymentType: Annotated[
        Literal["Full-time", "Self-employed", "Unemployed", "Part-time"],
        Field(..., description="Type of employment"),
    ]
    MaritalStatus: Annotated[
        Literal["Single", "Married", "Divorced"],
        Field(..., description="Marital status"),
    ]
    HasMortgage: Annotated[Literal["Yes", "No"], Field(..., description="Has mortgage or not")]
    HasDependents: Annotated[Literal["Yes", "No"], Field(..., description="Has dependents or not")]
    LoanPurpose: Annotated[
        Literal["Home", "Auto", "Education", "Business", "Other"],
        Field(..., description="Purpose of the loan"),
    ]
    HasCoSigner: Annotated[Literal["Yes", "No"], Field(..., description="Has co-signer or not")]

    # --- Computed Fields ---
    @computed_field
    @property
    def LoanPerCreditScore(self) -> float:
        return self.LoanAmount / (self.CreditScore + 1)  # avoid division by zero

    @computed_field
    @property
    def IncomePerCreditLine(self) -> float:
        return self.Income / (self.NumCreditLines + 1)

    @computed_field
    @property
    def InterestOverIncome(self) -> float:
        return self.InterestRate / self.Income if self.Income else 0

    @computed_field
    @property
    def MonthlyPayment(self) -> float:
        return self.LoanAmount / self.LoanTerm

    @computed_field
    @property
    def CreditScorePerAge(self) -> float:
        return self.CreditScore / self.Age

    @computed_field
    @property
    def EmploymentStability(self) -> float:
        return self.MonthsEmployed / self.Age

    @computed_field
    @property
    def DTI_LoanRatio(self) -> float:
        return self.DTIRatio / self.LoanAmount if self.LoanAmount else 0

    @computed_field
    @property
    def IncomePerLoanTerm(self) -> float:
        return self.Income / self.LoanTerm

    @computed_field
    @property
    def LTI(self) -> float:
        return self.LoanAmount / self.Income if self.Income else 0

    @computed_field
    @property
    def YearsEmployed(self) -> float:
        return self.MonthsEmployed / 12

    @computed_field
    @property
    def EmploymentToLoanTerm(self) -> float:
        return self.MonthsEmployed / self.LoanTerm
