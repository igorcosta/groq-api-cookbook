import dspy

class ExecutiveSummary(dspy.Signature):
    """Generate an executive summary with key points, trade-offs, and a conclusion."""
    text = dspy.InputField(desc="Input text describing machine learning in healthcare.")
    executive_summary = dspy.OutputField(desc="Executive summary of the text.")
    key_points = dspy.OutputField(desc="Key points extracted from the text.")
    trade_offs = dspy.OutputField(desc="Trade-offs extracted from the text.")
    conclusion = dspy.OutputField(desc="Conclusion drawn from the text.")

class ExecutiveSummaryPipeline(dspy.Program):
    class ExecutiveSummaryComponent(dspy.ChainOfThought):
        def __init__(self):
            super().__init__(signature=ExecutiveSummary, temperature=0.7, max_tokens=2048, max_depth=10)

    class KeyPointsComponent(dspy.ChainOfThought):
        def __init__(self):
            super().__init__(signature=ExecutiveSummary, temperature=0.7, max_tokens=1024, max_depth=10)

    class TradeOffsComponent(dspy.ChainOfThought):
        def __init__(self):
            super().__init__(signature=ExecutiveSummary, temperature=0.7, max_tokens=1024, max_depth=10)

    class ConclusionComponent(dspy.ChainOfThought):
        def __init__(self):
            super().__init__(signature=ExecutiveSummary, temperature=0.7, max_tokens=512, max_depth=10)

    def __init__(self):
        super().__init__()
        self.executive_summary = self.ExecutiveSummaryComponent()
        self.key_points = self.KeyPointsComponent()
        self.trade_offs = self.TradeOffsComponent()
        self.conclusion = self.ConclusionComponent()

    def forward(self, text):
        summary = self.executive_summary(text=text).executive_summary
        key_points = self.key_points(text=text).key_points
        trade_offs = self.trade_offs(text=text).trade_offs
        conclusion = self.conclusion(text=text).conclusion

        result = f"Executive Summary:\n{summary}\n\nKey Points:\n{key_points}\n\nTrade Offs:\n{trade_offs}\n\nConclusion:\n{conclusion}"
        return result
