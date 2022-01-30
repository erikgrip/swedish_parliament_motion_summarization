import pytest
import pandas as pd
import unittest

from fetch_data.src.clean_text import trim_whitespace, trim_motion_text



#@pytest.mark.parametrize("player1_points, player2_points, expected_score",
#                          [(0, 0, "Love-All"),
#                           (1, 1, "Fifteen-All"),
#                           (2, 2, "Thirty-All"),
#                           (3, 3, "Fourty-All")])
#def test_score_tennis(player1_points, player2_points, expected_score):
#    assert score_tennis(player1_points, player2_points) == expected_score


class PhoneBookTest(unittest.TestCase):

    # SetUp is a method inherited from unittest.TestCase
    # It is called before every test_method. There's also a tearDown
    # method in case of creating files, setting up db connection etc.
    def setUp(self) -> None:
        self.df = pd.DataFrame([
            {'id': 'GX02Sf368',
             'date': '2009-09-25 00:00:00',
             'title': 'Ökad valfrihet för föräldrar',
             'subtitle': 'av John Doe (p)',
             'text': 'Motion till riksdagen 2009/10:Sf368 av John Doe (p) Ökad valfrihet för föräldrar m1382 Förslag till riksdagsbeslut Riksdagen tillkännager för regeringen som sin mening vad som anförs i motionen om att föräldraförsäkringen bör ändras så att föräldrarna helt fritt får välja själva hur de vill använda sina 480 dagar. Motivering Föräldrapenning är den ersättning föräldrar får för att kunna vara hemma med sina barn i stället för att arbeta. Den betalas ut i sammanlagt 480 dagar per barn. Vid gemensam vårdnad har föräldrarna rätt till 240 dagar var med föräldrapenning. Av dessa är 60 dagar reserverade för var och en av föräldrarna. De övriga kan man välja att avstå till den andra föräldern. Som det är idag finns det familjer som inte själva får välja hur de vill planera sin vardag och tiden tillsammans med sina barn. Eftersom 60 dagar är reserverade för vardera föräldern, minskar utrymmet för familjerna att själva planera sin tid med barnen. Det är inte bra. Graden av valfrihet inom föräldraförsäkringen måste öka. Det finns ingen anledning för staten att styra familjer så pass, att vissa måste avstå från att spendera tid tillsammans med sina barn då 60 dagar är låsta till vardera föräldern. Alla familjer ser olika ut, och har olika behov, därför bör politiker så långt det är möjligt låta bli att styra familjepolitiken och istället skapa möjligheter för familjer att själva få pusslet i vardagen att gå ihop. Därför bör föräldraförsäkringen ändras så att föräldrarna helt fritt själva får välja hur de vill använda sina 480 dagar. Stockholm den 25 september 2009 John Doe (p)',
             'main_author': 'John Doe',
             'author_party': 'M'},
            {'id': 'GX02Ju219',
             'date': '2009-09-28 00:00:00',
             'title': 'Personalpool för polisen',
             'subtitle': 'av Hans Backman (fp)',
             'text': 'Motion till riksdagen 2009/10:Ju219 av Hans Backman (fp) Personalpool för polisen Förslag till riksdagsbeslut Förslag till riksdagsbeslut Riksdagen tillkännager för regeringen som sin mening vad som anförs i motionen om att polisen ska inrätta en rikstäckande personalpool som får rycka in och tjänstgöra på orter där det råder polisbrist. Motivering Trygghet är viktigt. Det är viktigt att polisorganisationen inte bara finns i större tätorter utan att det också finns polisiär närvaro på mindre orter. Jag vill ha fler poliser på våra gator och torg. Jag vill ha polisen tillbaka nära oss. Vardagsbrotten måste lösas och brottsoffren måste värnas. Bristen på synliga poliser på gator och torg i Sverige är sedan länge ett problem. Under de senaste åren har brottsligheten ökat och blivit grövre. Särskilt drabbade är de mindre orterna i Sverige. Polisen har en central roll i att ge skydd åt medborgarna. Fler och synligare poliser fungerar både brottsförebyggande och stärker polisens kapacitet. Fler brott upptäcks. Andelen uppklarade brott kommer att öka. Vi kan intensifiera arbetet mot ungdomsbrottslighet och samtidigt stävja den grova organiserade brottsligheten. I valet för tre år sedan lovade alliansen att det ska finnas 20 000 poliser år 2010 och det löftet säkerställer alliansregeringen nu genom ett betydande resurstillskott till polisen, vilket är både efterlängtat och mycket bra. Kommuninnevånarna i flera av Sveriges mindre kommuner känner ju idag oro för sin rättstrygghet. Företagarna känner oro för att utsättas för fler inbrott. Redan idag drabbas företagare som haft många inbrott av svårigheter när de ska försäkra sina firmor. Detta gäller bland annat min egen hemkommun Hofors. Det riskerar att leda till att befolkning i landets mindre kommuner flyttar till platser där de kan känna sig trygga. Kommunerna kan också förlora företag och arbetslösheten riskerar därmed att öka. På sikt vore detta givetvis förödande för många av Sveriges mindre kommuner. Närpolisreformen var rätt tänkt när den genomfördes. Tanken att polisen ska arbeta utifrån ett lokalt perspektiv och finnas bland medborgarna var en god tanke. Men de stora förändringarna i arbetssätt skedde samtidigt som omfattande neddragningar av såväl poliser som civilanställda skedde. Detta har bidragit till att reformen inte fått det genomslag vi hoppats på. Jag anser därför att polisen ska inrätta en rikstäckande personalpool som får rycka in och tjänstgöra på orter där det råder polisbrist. De poliser som vill delta i personalpoolen förbinder sig att på kort varsel flytta sin tjänstgöring till den ort som för tillfället behöver hjälp. De som ställer upp på att tjänstgöra i personalpoolen ska få extra lönetillägg och traktamente som är så väl tilltaget att det ekonomiskt lönar sig att tjänstgöra i kommuner med polisbrist. Att tjänstgöra i en sådan här personalpool skulle till exempel kunna vara attraktivt för unga poliser som ännu inte bildat familj och därför har lätt att snabbt byta arbetsplats. En rikstäckande personalpool som får rycka in och tjänstgöra på orter där det råder polisbrist skulle ge förutsättningar för att de positiva och viktiga satsningar på fler poliser som alliansregeringen nu gör kan få full effekt i Gävleborg. Stockholm den 28 september 2009 Hans Backman (fp)',
             'main_author': 'Hans Backman',
             'author_party': 'FP'},
            {'id': 'G5021417',
             'date': '1982-01-26 00:00:00',
             'title': 'Särskilt anslag för förstärkning av vinterberedskapen vid SJ',
             'subtitle': 'Lars Werner m. fl.',
             'text': 'Observera att dokumentet är inskannat och fel kan förekomma. 6 Motion 1981/82:1417 Lars Werner m. fl. Särskilt anslag för förstärkning av vinterberedskapen vid SJ Bakgrund Vänsterparitet kommunsiterna föreslog i motion 1980/81:34 att det av SJ till regeringen den 13 juni 1980 redovisade s. k. huvudalternativet till långsiktig investeringsplan för perioden 1981/82-1990/91 i princip skulle ligga till grund för SJ:s anslagsframställning. I 1980/81 års prisnivå skulle detta alternativ betyda 21 955 milj. kr. under berörd tioårsperiod. Men riksdagsmajoriteten beslutade en nedskärning härav till en med 5 % i reala termer ökad årlig ram under fem år, vilket innebär kraftiga sänkningar i jämförelse med nyssnämnda SJ:s huvudalternativ. I sammanhanget redovisade SJ en rad angelägna investeringsobjekt som därmed måste skjutas på framtiden. I denna motion föreslår nu vpk att riksdagen beslutar om ett extra investeringsanslag på 485 milj. kr. under en treårsperiod i syfte att stärka SJ:s vinterberedskap. Därmed ges riksdagen möjlighet att pröva denna anslagsfråga. Vintertrafikens problem Redan i november 1979 hemställde SJ om ett särskilt investeringsanslag på 390 milj. kr. fördelat på tre år och med syfte att skaffa resurser för snara och effektiva åtgärder. Det gällde att förhindra en upprepning av problemen vintern 1978/79. Kommunikationsminister Adelsohn kommenterade denna framställning i budgetpropositionen för 1980/81 med följande: Investeringsramen för budgetåret 1980/81 har beräknats för att ge utrymme åt investeringar för att stärka SJ:s vinterberedskap. Framställningen från SJ blev alltså avslagen. Vpk tog i motion 1979/80:924 upp frågan om ett särskilt anslag för att förstärka vinterberedskapen vid SJ. Motionen byggde på den tidigare nämnda skrivelsen från SJ och föreslog bl. a. att riksdagen skulle besluta att under en treårsperiod bevilja SJ ett anslag på 390 milj. kr. Denna motion blev avstyrkt av ett enigt utskott och alltså avslagen av riksdagen mot vpk:s röster. I den mån de kraftiga störningarna i vinterns tågtrafik sammanhänger med otillräckliga anslag för att anskaffa erforderlig materiel, anser sig vpk inte behöva dela ansvaret härför med övriga partier i riksdagen. Mot. 1981/82:1417 7 SJ:s resurser måste förstärkas Erfarenheterna från den gångna delen av vintern bekräftar nu vad många befarat: SJ:s resurser är helt otillräckliga för att klara ett ökat transportbehov under svåra vinterförhållanden. Det är inte vår avsikt att i denna motion analysera orsakerna till problemen med SJ:s vintertrafik, problem som gällt trafiken såväl med fjärrtåg som med lokaltåg/pendeltåg. Vi är väl medvetna om att bristande samarbete mellan SJ:s olika avdelningar liksom andra organisatoriska svagheter spelat en stor roll i sammanhanget. Vi är också på det klara med att personalens mycket uppoffrande insatser var avörande för att få pendeltågen att så långt möjligt rulla. Men det står också klart att en resursförstärkning i form av extra investeringsanslag är högst angelägen. Det gäller en rad arbeten: exempelvis måste pendeltågen litt. XI och bangården vid Stockholms Central byggas om, ny driftverkstad för personvagnar byggas vid Hagalund liksom ny signalanläggning därstädes. En ordentlig upprustning med snöröjningsmaskiner och redskap är angelägen liksom värmetält och infravärmeutrustning för avisning av fordon. Hemställan Med hänvisning till det anförda föreslås att riksdagen beslutar att under en treårsperiod bevilja statens järnvägar ett anslag på 485 000 000 kr. för förbättring av driftsäkerheten under vinterförhållanden, varvid början sker i 1982/83 års budget med ett anslag på 165 000 000 kr. Stockholm den 26 januari 1982 LARS WERNER (vpk) EIVOR MARKLUND (vpk) C.-H. HERMANSSON (vpk) NILS BERNDTSON (vpk) BERTIL MÅBRINK (vpk) EVA HJELMSTRÖM (vpk)',
             'main_author': 'LARS WERNER',
             'author_party': 'vpk'},
            {'id': 'GI02L7',
             'date': '1994-10-25 00:00:00',
             'title': 'med anledning av skr. 1994/95:30 Återkallelse av vissa propositioner',
             'subtitle': 'av Göran Hägglund m.fl. (kds, c, fp, m)',
             'text': 'Motion till riksdagen 1994/95:L7 av Göran Hägglund m.fl. (kds, m, c, fp) med anledning av skr. 1994/95:30 Återkallelse av vissa propositioner Regeringen har genom rubricerade skrivelse återkallat bl.a. proposition 1994/95:16 Den framtida konsumentpolitiken. De förslag som har framförts i propositionen är enligt vår mening angelägna. Riksdagen bör därför föreläggas förslagen för beslut. Hemställan Med hänvisning till motivtext och lagförslag i återkallad proposition 1994/95:16 hemställs att riksdagen godkänner propositionens förslag om konsumentpolitikens mål och inriktning (avsnitt 5). Stockholm den 24 oktober 1994 Göran Hägglund (kds) Lars Tobisson (m) Per-Ola Eriksson (c) Lars Leijonborg (fp)',
             'main_author': 'Göran Hägglund',
             'author_party': 'kds'},
            {'id': 'GX02So354',
             'date': '2009-09-30 00:00:00',
             'title': 'Service- och signalhundar',
             'subtitle': 'av Jan-Olof Larsson (s)',
             'text': 'Motion till riksdagen 2009/10:So354 av Jan-Olof Larsson (s) Service- och signalhundar s35016 Förslag till riksdagsbeslut Riksdagen tillkännager för regeringen som sin mening vad som anförs i motionen om vikten av service- och signalhundar. Motivering För människor med funktionshinder kan en hund, specialtränad och utbildad, vara till stor nytta i det dagliga livet. En ledarhund för en synskadad är ett kostnadsfritt hjälpmedel som utbildas centralt och bekostas med statliga medel. I dag blir det mer och mer vanligt att hundar även används för att hjälpa människor med funktionshinder. Dessa hundar kallas service- och signalhundar. En sådan hund kostar i inköp mellan 10 000 och 15 000 kronor. Träning av hunden sker sedan både i hemmet och på offentliga platser där hunden kommer att arbeta. Att utbilda hunden tar cirka ett år och kostar mellan 30 000 och 50 000 kronor. Utbildningarna genomförs till exempel genom Svenska service- och signalhundsförbundet och Nordiska assistanshundar. Båda organisationerna drivs ideellt av hundägare tillsammans med hundinspektörer. Dessa utbildningar bekostas av den organisation som utbildar och examinerar hunden. Inga statliga bidrag betalas ut för vare sig inköp eller utbildning, och därmed är organisationen beroende av penninggåvor, sponsorer, fonder och liknande för att ha råd med utbildningen. Eftersom en service- och signalhund arbetar på samma sätt som en ledarhund är det rimligt att inköp, träning och övriga kostnader hanteras lika för de båda grupperna av hundar. Båda arbetar med att underlätta vardagen för den som är funktionshindrad och utför ett oerhört viktigt och angeläget arbete. Jag yrkar på att åtgärder vidtas så att det blir en liktydig behandling när det gäller kostnader för inköp och träning av hundar som ska arbeta för människor med funktionshinder. Stockholm den 30 september 2009 Jan-Olof Larsson (s)',
             'main_author': 'Jan-Olof Larsson',
             'author_party': 'S'}
        ])

    def test_trim_whitespace_removes_leading(self):
        self.s = pd.Series([
            'string to be trimmed',
            ' string to be trimmed',
            '  string to be trimmed'])
        s2 = trim_whitespace(self.s)
        self.assertTrue('string to be trimmed' == s2.values.all())


    def test_trim_whitespace_removes_trailing(self):
        self.s = pd.Series([
            'string to be trimmed',
            'string to be trimmed ',
            'string to be trimmed  '])
        s2 = trim_whitespace(self.s)
        self.assertTrue('string to be trimmed' == s2.values.all())


    def test_trim_whitespace_removes_duplicate_in_middle(self):
        self.s = pd.Series([
            'string to be trimmed',
            'string to  be trimmed',
            'string to be   trimmed'])
        s2 = trim_whitespace(self.s)
        self.assertTrue('string to be trimmed' == s2.values.all())


    def test_trim_motion_text_remove_up_and_including_subtitle(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'A motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'Some leading text av John Doe (p) the motion text starts here',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, 'the motion text starts here')


    def test_trim_motion_text_with_no_subtitle(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'A motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'Some leading text the motion text starts here',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, self.df['text'].values[0])


    def test_trim_motion_text_remove_leading_title_and_subtitle(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'a motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'av John Doe (p) a motion title the motion text starts here and then a motion title again.',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, 'the motion text starts here and then a motion title again.')


    def test_trim_motion_text_remove_leadning_title_no_subtitle(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'a motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'Something else a motion title the motion text starts here and then a motion title again.',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, self.df['text'].values[0])


    def test_trim_motion_text_remove_up_and_including_recommendation_formulation_1(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'a motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'some text förslag till riksdagsbeslut riksdagen the recommendation. the motion text starts here',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, 'the motion text starts here')


    def test_trim_motion_text_remove_up_and_including_recommendation_formulation_2(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'a motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'some text riksdagen tillkännager för regeringen som sin mening the recommendation. the motion text starts here',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, 'the motion text starts here')


    def test_trim_motion_text_remove_up_and_including_recommendation_formulation_3(self):
        self.df = pd.DataFrame([{
            'id': 'GX02Sf368',
            'date': '2009-09-25 00:00:00',
            'title': 'a motion title',
            'subtitle': 'av John Doe (p)',
            'text': 'some text riksdagen ställer sig bakom det som anförs the recommendation. the motion text starts here',
            'main_author': 'John Doe',
            'author_party': 'P'}
        ])
        s = trim_motion_text(self.df.iloc[0, :])
        self.assertEqual(s, 'the motion text starts here')





    #def test_missing_name(self):
    #    with self.assertRaises(KeyError):
    #        self.phonebook.lookup('missing')
    #
    #def test_consistent_when_empty(self):
    #    self.assertTrue(self.phonebook.is_consistent())
    #
    #def test_consistent_with_diffrent_entries(self):
    #    self.phonebook.add('Bob', '12345')
    #    self.phonebook.add('Sue', '012345')
    #    self.assertTrue(self.phonebook.is_consistent())
    #
    #def test_inconsistent_with_duplicate_entries(self):
    #    self.phonebook.add('Bob', '12345')
    #    self.phonebook.add('Sue', '12345')
    #    self.assertFalse(self.phonebook.is_consistent())
    #
    #def test_inconsistent_with_duplicate_prefix(self):
    #    self.phonebook.add('Bob', '12345')
    #    self.phonebook.add('Ann', '123')
    #    self.assertFalse(self.phonebook.is_consistent())


if __name__ == '__main__':
    unittest.main()
