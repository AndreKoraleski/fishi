from fishi.woodscape import classes


def test_class_taxonomy():
    assert len(classes.CLASS_NAMES) == 10
    assert classes.CLASS_NAMES[classes.VOID_ID] == "void"
    assert classes.VOID_ID not in classes.PROMPTS
    assert len(classes.PROMPTS) == 9
    assert classes.PROMPTS[9] == "traffic sign"  # underscores become spaces
