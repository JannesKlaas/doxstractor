
import re

def parseNumber(text):
    """
        Return the first number in the given text for any locale.
        TODO we actually don't take into account spaces for only
        3-digited numbers (like "1 000") so, for now, "1 0" is 10.
        TODO parse cases like "125,000.1,0.2" (125000.1).

        :example:
        >>> parseNumber("a 125,00 €")
        125
        >>> parseNumber("100.000,000")
        100000
        >>> parseNumber("100 000,000")
        100000
        >>> parseNumber("100,000,000")
        100000000
        >>> parseNumber("100 000 000")
        100000000
        >>> parseNumber("100.001 001")
        100.001
        >>> parseNumber("$.3")
        0.3
        >>> parseNumber(".003")
        0.003
        >>> parseNumber(".003 55")
        0.003
        >>> parseNumber("3 005")
        3005
        >>> parseNumber("1.190,00 €")
        1190
        >>> parseNumber("1190,00 €")
        1190
        >>> parseNumber("1,190.00 €")
        1190
        >>> parseNumber("$1190.00")
        1190
        >>> parseNumber("$1 190.99")
        1190.99
        >>> parseNumber("$-1 190.99")
        -1190.99
        >>> parseNumber("1 000 000.3")
        1000000.3
        >>> parseNumber('-151.744122')
        -151.744122
        >>> parseNumber('-1')
        -1
        >>> parseNumber("1 0002,1.2")
        10002.1
        >>> parseNumber("")

        >>> parseNumber(None)

        >>> parseNumber(1)
        1
        >>> parseNumber(1.1)
        1.1
        >>> parseNumber("rrr1,.2o")
        1
        >>> parseNumber("rrr1rrr")
        1
        >>> parseNumber("rrr ,.o")

    """
    try:
        # First we return None if we don't have something in the text:
        if text is None:
            return None
        if isinstance(text, int) or isinstance(text, float):
            return text
        text = text.strip()
        if text == "":
            return None
        # Next we get the first "[0-9,. ]+":
        n = re.search("-?[0-9]*([,. ]?[0-9]+)+", text).group(0)
        n = n.strip()
        if not re.match(".*[0-9]+.*", text):
            return None
        # Then we cut to keep only 2 symbols:
        while " " in n and "," in n and "." in n:
            index = max(n.rfind(','), n.rfind(' '), n.rfind('.'))
            n = n[0:index]
        n = n.strip()
        # We count the number of symbols:
        symbolsCount = 0
        for current in [" ", ",", "."]:
            if current in n:
                symbolsCount += 1
        # If we don't have any symbol, we do nothing:
        if symbolsCount == 0:
            pass
        # With one symbol:
        elif symbolsCount == 1:
            # If this is a space, we just remove all:
            if " " in n:
                n = n.replace(" ", "")
            # Else we set it as a "." if one occurence, or remove it:
            else:
                theSymbol = "," if "," in n else "."
                if n.count(theSymbol) > 1:
                    n = n.replace(theSymbol, "")
                else:
                    n = n.replace(theSymbol, ".")
        else:
            # Now replace symbols so the right symbol is "." and all left are "":
            rightSymbolIndex = max(n.rfind(','), n.rfind(' '), n.rfind('.'))
            rightSymbol = n[rightSymbolIndex:rightSymbolIndex+1]
            if rightSymbol == " ":
                return parseNumber(n.replace(" ", "_"))
            n = n.replace(rightSymbol, "R")
            leftSymbolIndex = max(n.rfind(','), n.rfind(' '), n.rfind('.'))
            leftSymbol = n[leftSymbolIndex:leftSymbolIndex+1]
            n = n.replace(leftSymbol, "L")
            n = n.replace("L", "")
            n = n.replace("R", ".")
        # And we cast the text to float or int:
        n = float(n)
        if n.is_integer():
            return int(n)
        else:
            return n
    except: pass
    return None

def most_common(lst):
    return max(set(lst), key=lst.count)