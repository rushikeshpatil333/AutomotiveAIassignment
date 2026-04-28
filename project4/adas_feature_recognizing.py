L0 = ["BSW","LDW","FCW","Reverse Camera"]
L1 = ["ACC","LKA","AEB","ESC","TSR"]
L2 = ["LCA","TJA","PA"]
L3 = ["HP","DMS","ALC","ODA","APA"]

print("Level 4 and level 5 have been considered to have all fetures hence pls enter the value as Self Driving")

a = input("Enter the ADAS feature: ")

if a in L0:
  print("ADAS Level 0")
elif a in L1:
  print("ADAS Level 1")
elif a in L2:
  print("ADAS Level 2")
elif a in L3:
  print("ADAS Level 3")
elif a == "Self Driving":
    x = input("Restricted or Free")
    if x == "Restricted":
      print("ADAS Level 4")
    else:
      print("ADAS Level 5")
