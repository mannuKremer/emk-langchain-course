import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    print("Hello from emk-langchain-course!")
    print(os.environ.get("OPENAI_API_KEY"))
    information = """
    Elon Musk

Article
Talk
Read
View source
View history

Appearance hide
Text

Small

Standard

Large
Width

Standard

Wide
Color

Automatic

Light

Dark
Page extended-confirmed-protected
From Wikipedia, the free encyclopedia
For other uses, see Elon Musk (disambiguation).
Elon Musk
FRS

Musk in 2025
Born	Elon Reeve Musk
June 28, 1971 (age 55)
Pretoria, Transvaal,[a] South Africa
Citizenship	
South Africa
Canada
United States (since 2002)
Education	University of Pennsylvania (BA, BS)
Occupations	
CEO and product architect of Tesla
Founder, CEO, and chief engineer of SpaceX
Founder and CEO of SpaceXAI
Founder and CTO of X Corp.
Co-founder of Neuralink, The Boring Company, OpenAI, Zip2, X.com and PayPal
President of the Musk Foundation
Political party	Independent[b]
Spouses	
Justine Wilson
​
​(m. 2000; div. 2008)​
Talulah Riley
​
​(m. 2010; div. 2012)​
​
​(m. 2013; div. 2016)​
Partners	
Amber Heard (2017–2018)
Grimes (2018–2021)
Ashley St. Clair (2023–2025)
Shivon Zilis (until 2026)
Children	At least 14,[c] including Vivian Wilson
Parents	
Errol Musk (father)
Maye Musk (mother)
Relatives	Musk family
Awards	Full list
Musk's voice
Duration: 3 minutes and 10 seconds.3:10
Musk on his main goals for SpaceX and the future of human civilization
Recorded January 22, 2026
Signature

	
This article is part of
a series about
Elon Musk
Personal
Companies
Politics
In the arts and media
vte
Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, technologist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company. Musk joined Tesla as an angel investor in 2004 and became its CEO and product architect in 2008. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.

Musk was the largest donor in the 2024 United States presidential election, in which he supported Republican nominee Donald Trump. After Trump was inaugurated in January 2025, Musk was an advisor to President Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump. Musk also became the largest political donor in American history in 2026, in which he is supporting Republican nominees in the midterm elections.

Musk is a supporter of global far-right politics, figures, and political parties. His political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including spreading COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, white nationalist, racist, and transphobic comments. His acquisition of Twitter was controversial because, following his pledge to decrease censorship, there was an increase in hate speech and misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE and its cuts to the US Agency for International Development (USAID).

Early life and education
See also: Musk family
Elon Reeve Musk was born on June 28, 1971, in Pretoria, Transvaal Province (Gauteng Province since 1994) South Africa's administrative capital.[3][4] He is of British and Pennsylvania Dutch ancestry.[5][6] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[7][8][9] Musk therefore holds both South African and Canadian citizenship from birth.[10] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[11][12][13][14]

His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Musk was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the technocracy movement[15][16] who moved to South Africa in 1950.[17] Haldeman's anti-government, anti-democratic and conspiracist views, which included the promotion of far-right antisemitic conspiracy theories,[18][19] "fanatical" support of apartheid,[19] and according to Errol, support of Nazism,[17] have been suggested as an influence on Musk.[20][21][22][23] During his childhood, Musk was told stories by his grandmother of Haldeman's travels and exploits, and he has suggested that all of Haldeman's descendants have his "desire for adventure, exploration – doing crazy things".[24]

Musk has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[25][26][9][27] He was baptized as a child in the Anglican Church of Southern Africa.[28][29] The Musk family was wealthy during his youth.[14] Despite both Musk and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[14] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[30][31] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[3]

After his parents divorced in 1979, Musk, aged around 9, chose to live with his father because he had an Encyclopædia Britannica set and a computer.[32][5][11] He later regretted his decision and became estranged from his father.[33] Musk has recounted trips to a wilderness school that he described as a "paramilitary Lord of the Flies" where "bullying was a virtue" and children were encouraged to fight over rations.[34] In one incident, after an altercation with a fellow pupil, Musk was thrown down concrete steps and beaten severely, leading to him being hospitalized for his injuries.[35] He described his father berating him after he was discharged from the hospital.[35] Errol denied berating Musk and claimed, "The [other] boy had just lost his father to suicide, and Musk had called him stupid. He had a tendency to call people stupid. How could I possibly blame that child?"[36]

Musk was an enthusiastic reader of books, and had attributed his success in part to having read The Lord of the Rings, the Foundation series, and The Hitchhiker's Guide to the Galaxy.[13][37] At age ten, he developed an interest in computing and video games, teaching himself how to program from the VIC-20 user manual.[38] At age twelve, in 1983, Musk sold his BASIC-based game Blastar to PC and Office Technology magazine for approximately $500 (equivalent to $1,600 in 2025).[39][40]

"""

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    #llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.text)


if __name__ == "__main__":
    main()
