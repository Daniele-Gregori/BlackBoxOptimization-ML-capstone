from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl

session = WolframLanguageSession()  # uses default Mathematica path
#result = session.evaluate(wl.Integrate(wl.Sin(wl.x), wl.x))
result = session.evaluate(wl.LLMSynthesize(
    "Hi who are you?",
    wl.Rule(wl.LLMEvaluator, wl.Rule("Model", wl.Association(
        wl.Rule("Service", "Anthropic"),
        wl.Rule("Name", "claude-opus-4-5")
    )))
))
print(result)
session.terminate()