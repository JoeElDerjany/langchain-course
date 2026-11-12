from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """
    Jesus[e] (c. 6 to 4 BC – AD 30 or 33), also known as Jesus Christ,[f] Jesus of Nazareth, and by various other names and titles, was a Jewish preacher and religious leader active in the Roman province of Judaea during the 1st century.[12] He is the central figure of Christianity, the world's largest religion. Most branches of Christianity consider Jesus the incarnation of God the Son and the awaited Messiah or Christ. Accounts of Jesus's life are contained in the Gospels, especially the four canonical Gospels of the New Testament. Since the Enlightenment, academic research has produced various views on the historical reliability of the Gospels and the extent to which they reflect the historical Jesus,[13][g][16][17] but virtually all modern scholars of antiquity agree that Jesus existed historically.[h]

    According to Christian tradition, as represented in the Gospels and the Acts of the Apostles, Jesus was circumcised at eight days old, presented at the Temple in Jerusalem at 40 days. He was baptized by John the Baptist as a young adult, and after 40 days and nights of fasting in the Judean wilderness, he began his public ministry. He was an itinerant teacher whom his followers believed to possess divine authority in interpreting Jewish law. Jesus often debated with other Jews, most commonly the Pharisees and Sadducees, about how best to follow God; engaged in healings; taught in parables; and gathered followers, 12 of whom he appointed as his apostles.

    According to the New Testament accounts, Jesus was arrested in Jerusalem and tried by the Sanhedrin;[25] he was then handed over to the Roman authorities, and crucified on the order of Pontius Pilate, the Roman prefect of Judaea. In Christian theology, the Resurrection is the event in which Jesus rose from the dead, on the third day after his death, and this belief became the foundation of the early Christian Church, which expanded into a worldwide movement.[26]

    Christian theology includes the beliefs that Jesus was conceived by the Holy Spirit, was born of a virgin named Mary, performed miracles, founded the Christian Church, died by crucifixion as a sacrifice for atonement for sin, was resurrected by God the Father, and ascended into Heaven, from where he will return. Christians commonly believe that Jesus enables people to be reconciled to God. The Nicene Creed asserts that Jesus will judge the living and the dead, either before or after their bodily resurrection, an event associated with the Second Coming of Jesus in Christian eschatology. Many branches of Christianity consider Jesus as the incarnation of God the Son, the second of the three persons of the Trinity.[i]

    The birth of Jesus in the Judean town of Bethlehem is celebrated annually, generally on 25 December,[j] as Christmas. His crucifixion is commemorated on Good Friday and his resurrection on Easter Sunday. The world's most widely used calendar era—in which the current year is AD 2026 (or 2026 CE)—is traditionally based on the approximate date of the birth of Jesus.[27]

    Mainstream Judaism rejects the belief that Jesus was the awaited messiah, holding that he did not fulfil messianic prophecies, was not lawfully anointed, and was neither divine nor resurrected. In contrast, Jesus in Islam[k] is considered the messiah and a prophet of God, who was sent to the Israelites and will return to Earth before the Day of Judgement. Muslims believe that Jesus was born of the virgin Mary but was neither God nor the son of God. Most Muslims do not believe that he was killed or crucified, but that God raised him into Heaven while he was still alive.[l] Jesus is also revered in the Baháʼí and Druze faiths, as well as in Rastafari.
    """

    summary_template = """
    Given the information {information} about a person, create:
    1. A short summary
    2. 2 interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-5")

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
