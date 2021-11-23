from bot import AI

ai = AI()

result = ai.processMessage("i like green")
print(result)


result = ai.processMessage("What color do I like?")
print(result)

print(str(ai.memory))