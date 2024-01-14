std::vector<CPolyline> polylines;
{
    CPolyline polyline;
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(100, 23.750000002069605), float64_t2(100.15719746347405, 23.401622646660716), float64_t2(100.23856885839848, 23.01305819440771) });
        quadraticBeziers.push_back({ float64_t2(100.23856885839848, 23.01305819440771), float64_t2(100.31994025332291, 22.624493745604045), float64_t2(100.31994025601874, 22.222222225671565) });
        quadraticBeziers.push_back({ float64_t2(100.31994025601874, 22.222222225671565), float64_t2(100.31994025601874, 21.819950702289738), float64_t2(100.23856886109431, 21.431386253486075) });
        quadraticBeziers.push_back({ float64_t2(100.23856886109431, 21.431386253486075), float64_t2(100.15719746616986, 21.04282180123307), float64_t2(100.00000000269583, 20.694444445824182) });
        quadraticBeziers.push_back({ float64_t2(100.00000000269583, 20.694444445824182), float64_t2(99.68005974937293, 19.985400312752635), float64_t2(99.12590697330299, 19.576033491089387) });
        quadraticBeziers.push_back({ float64_t2(99.12590697330299, 19.576033491089387), float64_t2(98.57175419723305, 19.166666669426142), float64_t2(97.93187369058725, 19.1666666659768) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 97.93187369058725, 19.1666666659768 });
        linePoints.push_back({ 78.85003878081946, 19.1666666659768 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(78.85003878081946, 19.1666666659768), float64_t2(79.22051635362205, 15.577974339464195), float64_t2(80.4233657089217, 12.301586506267387) });
        quadraticBeziers.push_back({ float64_t2(80.4233657089217, 12.301586506267387), float64_t2(81.62621506422133, 9.025198669621238), float64_t2(83.54835937178198, 6.369120275808705) });
        quadraticBeziers.push_back({ float64_t2(83.54835937178198, 6.369120275808705), float64_t2(85.47050367664683, 3.713041878546829), float64_t2(87.931247000807, 1.926964155777737) });
        quadraticBeziers.push_back({ float64_t2(87.931247000807, 1.926964155777737), float64_t2(90.39199032227135, 0.14088642955930145), float64_t2(93.16000438794643, -0.6072859659239098) });
        quadraticBeziers.push_back({ float64_t2(93.16000438794643, -0.6072859659239098), float64_t2(95.92801845631732, -1.3554583614071212), float64_t2(98.74308924910814, -0.9953916279806031) });
        quadraticBeziers.push_back({ float64_t2(98.74308924910814, -0.9953916279806031), float64_t2(101.55816004189894, -0.6353248980034281), float64_t2(104.1556498618174, 0.7991320005169621) });
        quadraticBeziers.push_back({ float64_t2(104.1556498618174, 0.7991320005169621), float64_t2(106.75313967904, 2.2335888955880097), float64_t2(108.88886506116053, 4.607586279787399) });
        quadraticBeziers.push_back({ float64_t2(108.88886506116053, 4.607586279787399), float64_t2(111.02459044058521, 6.981583667436133), float64_t2(112.49777722864498, 10.071948039586898) });
        quadraticBeziers.push_back({ float64_t2(112.49777722864498, 10.071948039586898), float64_t2(113.97096401940058, 13.16231241173766), float64_t2(114.64312164190477, 16.678526414627278) });
        quadraticBeziers.push_back({ float64_t2(114.64312164190477, 16.678526414627278), float64_t2(115.31527926710481, 20.194740417516893), float64_t2(115.12321988219698, 23.806253644741243) });
        quadraticBeziers.push_back({ float64_t2(115.12321988219698, 23.806253644741243), float64_t2(114.93116049998497, 27.417766871965593), float64_t2(113.8929391289186, 30.785070076860766) });
        quadraticBeziers.push_back({ float64_t2(113.8929391289186, 30.785070076860766), float64_t2(112.85471775515639, 34.152373281755935), float64_t2(111.06793496574123, 36.95891478823291) });
        quadraticBeziers.push_back({ float64_t2(111.06793496574123, 36.95891478823291), float64_t2(109.28115217632607, 39.765456294709885), float64_t2(106.91377891222056, 41.747400219793676) });
        quadraticBeziers.push_back({ float64_t2(106.91377891222056, 41.747400219793676), float64_t2(104.5464056454192, 43.729344141428115), float64_t2(101.82099269185252, 44.70037293654901) });
        quadraticBeziers.push_back({ float64_t2(101.82099269185252, 44.70037293654901), float64_t2(99.09557973559, 45.67140172822056), float64_t2(96.2683362887578, 45.54023142490122) });
        quadraticBeziers.push_back({ float64_t2(96.2683362887578, 45.54023142490122), float64_t2(93.44109283922978, 45.409061118132534), float64_t2(90.77780091723362, 44.18802270665765) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 90.77780091723362, 44.18802270665765 });
        linePoints.push_back({ 100, 23.750000002069605 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    polylines.push_back(polyline);
}
{
    CPolyline polyline;
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(85.13745643750717, 40.02135603999098), float64_t2(82.54736844392633, 37.21217846635867), float64_t2(80.92102223131728, 33.39850424133517) });
        quadraticBeziers.push_back({ float64_t2(80.92102223131728, 33.39850424133517), float64_t2(79.29467601870823, 29.58483001631167), float64_t2(78.85003878081946, 25.277777778467648) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 78.85003878081946, 25.277777778467648 });
        linePoints.push_back({ 91.79016525812575, 25.277777778467648 });
        linePoints.push_back({ 85.13745643750717, 40.02135603999098 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    polylines.push_back(polyline);
}
{
    CPolyline polyline;
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(57.85169294647871, 31.111111112490846), float64_t2(56.46982829835856, 31.111111112490846), float64_t2(55.19315183025195, 30.43448473499329) });
        quadraticBeziers.push_back({ float64_t2(55.19315183025195, 30.43448473499329), float64_t2(53.916475364841176, 29.75785835749573), float64_t2(52.93934950170601, 28.50761583281888) });
        quadraticBeziers.push_back({ float64_t2(52.93934950170601, 28.50761583281888), float64_t2(51.96222363857086, 27.257373311591373), float64_t2(51.433406931486545, 25.62385273545428) });
        quadraticBeziers.push_back({ float64_t2(51.433406931486545, 25.62385273545428), float64_t2(50.904590224402234, 23.990332155867858), float64_t2(50.904590224402234, 22.222222225671565) });
        quadraticBeziers.push_back({ float64_t2(50.904590224402234, 22.222222225671565), float64_t2(50.904590224402234, 20.454112295475273), float64_t2(51.433406931486545, 18.820591715888845) });
        quadraticBeziers.push_back({ float64_t2(51.433406931486545, 18.820591715888845), float64_t2(51.96222363857086, 17.18707113975176), float64_t2(52.93934950170601, 15.93682861852425) });
        quadraticBeziers.push_back({ float64_t2(52.93934950170601, 15.93682861852425), float64_t2(53.916475364841176, 14.686586093847398), float64_t2(55.19315183025195, 14.00995971634984) });
        quadraticBeziers.push_back({ float64_t2(55.19315183025195, 14.00995971634984), float64_t2(56.46982829835856, 13.333333338852283), float64_t2(57.85169294647871, 13.333333331953595) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(57.85169294647871, 13.333333331953595), float64_t2(59.233557597294684, 13.333333331953595), float64_t2(60.51023406540129, 14.009959709451156) });
        quadraticBeziers.push_back({ float64_t2(60.51023406540129, 14.009959709451156), float64_t2(61.7869105335079, 14.686586086948713), float64_t2(62.764036399338885, 15.936828611625565) });
        quadraticBeziers.push_back({ float64_t2(62.764036399338885, 15.936828611625565), float64_t2(63.74116226247405, 17.187071132853074), float64_t2(64.2699789722542, 18.8205917124395) });
        quadraticBeziers.push_back({ float64_t2(64.2699789722542, 18.8205917124395), float64_t2(64.79879567933851, 20.45411229202593), float64_t2(64.79879567933851, 22.222222225671565) });
        quadraticBeziers.push_back({ float64_t2(64.79879567933851, 22.222222225671565), float64_t2(64.79879567933851, 23.9903321593172), float64_t2(64.2699789722542, 25.62385273890363) });
        quadraticBeziers.push_back({ float64_t2(64.2699789722542, 25.62385273890363), float64_t2(63.74116226247405, 27.257373318490053), float64_t2(62.764036399338885, 28.507615839717566) });
        quadraticBeziers.push_back({ float64_t2(62.764036399338885, 28.507615839717566), float64_t2(61.7869105335079, 29.75785836439442), float64_t2(60.51023406540129, 30.434484741891975) });
        quadraticBeziers.push_back({ float64_t2(60.51023406540129, 30.434484741891975), float64_t2(59.233557597294684, 31.111111119389534), float64_t2(57.85169294647871, 31.111111112490846) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    polylines.push_back(polyline);
}
{
    CPolyline polyline;
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 64.453331546595, 33.24727201834321 });
        linePoints.push_back({ 73.6329219204022, 65.5174373511087 });
        linePoints.push_back({ 81.18532078502435, 48.779986981578446 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(81.18532078502435, 48.779986981578446), float64_t2(76.8151114804778, 44.53859427529905), float64_t2(74.15255625891092, 38.40257201205801) });
        quadraticBeziers.push_back({ float64_t2(74.15255625891092, 38.40257201205801), float64_t2(71.49000103734406, 32.26654974881698), float64_t2(70.98719450787826, 25.277777778467648) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 70.98719450787826, 25.277777778467648 });
        linePoints.push_back({ 68.4405957893852, 25.277777778467648 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(68.4405957893852, 25.277777778467648), float64_t2(68.02130531214648, 27.65660715392894), float64_t2(66.9937599858567, 29.71040044600765) });
        quadraticBeziers.push_back({ float64_t2(66.9937599858567, 29.71040044600765), float64_t2(65.9662146595669, 31.764193738086355), float64_t2(64.453331546595, 33.24727201834321) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    polylines.push_back(polyline);
}
{
    CPolyline polyline;
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 66.00744729793786, 4.444444445134313 });
        linePoints.push_back({ 63.77126705102386, 10.580340814259317 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(63.77126705102386, 10.580340814259317), float64_t2(65.54989424097649, 12.060947585161086), float64_t2(66.76183367848054, 14.289556343660312) });
        quadraticBeziers.push_back({ float64_t2(66.76183367848054, 14.289556343660312), float64_t2(67.9737731159846, 16.51816510560888), float64_t2(68.4405957893852, 19.1666666659768) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 68.4405957893852, 19.1666666659768 });
        linePoints.push_back({ 70.98719450787826, 19.1666666659768 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(70.98719450787826, 19.1666666659768), float64_t2(71.36483930774644, 13.91758321336022), float64_t2(72.98989932067414, 9.073726201636925) });
        quadraticBeziers.push_back({ float64_t2(72.98989932067414, 9.073726201636925), float64_t2(74.61495933360186, 4.229869193362969), float64_t2(77.32884433294333, 0.26395248456133735) });
        quadraticBeziers.push_back({ float64_t2(77.32884433294333, 0.26395248456133735), float64_t2(80.04272933228482, -3.701964224240294), float64_t2(83.58059024605699, -6.402905306054486) });
        quadraticBeziers.push_back({ float64_t2(83.58059024605699, -6.402905306054486), float64_t2(87.11845115982914, -9.10384639131802), float64_t2(91.13502678663764, -10.276225978439605) });
        quadraticBeziers.push_back({ float64_t2(91.13502678663764, -10.276225978439605), float64_t2(95.15160241614198, -11.44860556556119), float64_t2(99.25491361251649, -10.978010686597338) });
        quadraticBeziers.push_back({ float64_t2(99.25491361251649, -10.978010686597338), float64_t2(103.35822480889101, -10.507415807633489), float64_t2(107.1478278681378, -8.439771993154729) });
        quadraticBeziers.push_back({ float64_t2(107.1478278681378, -8.439771993154729), float64_t2(110.93743092468873, -6.372128178675969), float64_t2(114.0434970364382, -2.909217578255468) });
        quadraticBeziers.push_back({ float64_t2(114.0434970364382, -2.909217578255468), float64_t2(117.14956314818762, 0.5536930187156907), float64_t2(119.26897013621107, 5.073923634848109) });
        quadraticBeziers.push_back({ float64_t2(119.26897013621107, 5.073923634848109), float64_t2(121.38837712153871, 9.594154250980527), float64_t2(122.31429124654079, 14.730573856030349) });
        quadraticBeziers.push_back({ float64_t2(122.31429124654079, 14.730573856030349), float64_t2(122.77438956932917, 17.28292506671062), float64_t2(122.91726014586868, 19.895901831073893) });
        quadraticBeziers.push_back({ float64_t2(122.91726014586868, 19.895901831073893), float64_t2(123.06013071971236, 22.50887859198782), float64_t2(122.88226632552792, 25.118336898999083) });
        quadraticBeziers.push_back({ float64_t2(122.88226632552792, 25.118336898999083), float64_t2(122.52432727681717, 30.36968034036733), float64_t2(120.91746646086277, 35.22346700645155) });
        quadraticBeziers.push_back({ float64_t2(120.91746646086277, 35.22346700645155), float64_t2(119.31060564760418, 40.077253672535775), float64_t2(116.61163722658867, 44.059800670516715) });
        quadraticBeziers.push_back({ float64_t2(116.61163722658867, 44.059800670516715), float64_t2(113.912668808269, 48.04234767194699), float64_t2(110.3849861406619, 50.76499672054693) });
        quadraticBeziers.push_back({ float64_t2(110.3849861406619, 50.76499672054693), float64_t2(106.8573034730548, 53.4876457725962), float64_t2(102.84517445717923, 54.684692514301446) });
        quadraticBeziers.push_back({ float64_t2(102.84517445717923, 54.684692514301446), float64_t2(98.83304544130367, 55.8817392560067), float64_t2(94.72801527546359, 55.43636344028292) });
        quadraticBeziers.push_back({ float64_t2(94.72801527546359, 55.43636344028292), float64_t2(90.62298511231933, 54.99098762455914), float64_t2(86.82566526205497, 52.94665364824511) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 86.82566526205497, 52.94665364824511 });
        linePoints.push_back({ 76.54077671044429, 75.73978399374971 });
        linePoints.push_back({ 78.54280859807695, 82.77777777984738 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(78.54280859807695, 82.77777777984738), float64_t2(79.76121379402828, 82.90387485866194), float64_t2(80.89133992526189, 83.49996523034794) });
        quadraticBeziers.push_back({ float64_t2(80.89133992526189, 83.49996523034794), float64_t2(82.0214660564955, 84.09605559858458), float64_t2(82.95537027313219, 85.10520429246955) });
        quadraticBeziers.push_back({ float64_t2(82.95537027313219, 85.10520429246955), float64_t2(83.88927449246472, 86.11435298635452), float64_t2(84.53775588509906, 87.44017218963968) });
        quadraticBeziers.push_back({ float64_t2(84.53775588509906, 87.44017218963968), float64_t2(85.18623727773338, 88.76599139637418), float64_t2(85.48735681032304, 90.28184683993459) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(85.48735681032304, 90.28184683993459), float64_t2(85.51179122369678, 90.4048512827743), float64_t2(85.51391749273844, 90.53173758848398) });
        quadraticBeziers.push_back({ float64_t2(85.51391749273844, 90.53173758848398), float64_t2(85.51604375908428, 90.65862389074432), float64_t2(85.49575234410645, 90.78285534762674) });
        quadraticBeziers.push_back({ float64_t2(85.49575234410645, 90.78285534762674), float64_t2(85.45463302497751, 91.03460283605037), float64_t2(85.33124366116344, 91.23762853581596) });
        quadraticBeziers.push_back({ float64_t2(85.33124366116344, 91.23762853581596), float64_t2(85.2078542973494, 91.44065423558155), float64_t2(85.027293665429, 91.55366044767477) });
        quadraticBeziers.push_back({ float64_t2(85.027293665429, 91.55366044767477), float64_t2(84.8467330335086, 91.66666665976798), float64_t2(84.64572918977726, 91.66666666666666) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 84.64572918977726, 91.66666666666666 });
        linePoints.push_back({ 62.19363215047233, 91.66666666666666 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(62.19363215047233, 91.66666666666666), float64_t2(61.7924279822978, 91.66666666666666), float64_t2(61.41772310438694, 91.48319615741019) });
        quadraticBeziers.push_back({ float64_t2(61.41772310438694, 91.48319615741019), float64_t2(61.04301822378022, 91.29972564815371), float64_t2(60.74431068323698, 90.9570208578198) });
        quadraticBeziers.push_back({ float64_t2(60.74431068323698, 90.9570208578198), float64_t2(60.44560314538957, 90.61431607093525), float64_t2(60.26235184262354, 90.15764787931133) });
        quadraticBeziers.push_back({ float64_t2(60.26235184262354, 90.15764787931133), float64_t2(60.07910054255334, 89.70097968768742), float64_t2(60.0355127483339, 89.19067340040648) });
        quadraticBeziers.push_back({ float64_t2(60.0355127483339, 89.19067340040648), float64_t2(60.01390181859936, 88.93766236594982), float64_t2(60.02858552298495, 88.6838392044107) });
        quadraticBeziers.push_back({ float64_t2(60.02858552298495, 88.6838392044107), float64_t2(60.04326922467471, 88.4300160428716), float64_t2(60.09375856376019, 88.18383356211362) });
        quadraticBeziers.push_back({ float64_t2(60.09375856376019, 88.18383356211362), float64_t2(60.195592173405934, 87.68730000765235), float64_t2(60.42939508894582, 87.27013066204059) });
        quadraticBeziers.push_back({ float64_t2(60.42939508894582, 87.27013066204059), float64_t2(60.66319800718152, 86.85296131297946), float64_t2(60.99808515344623, 86.57026372642981) });
        quadraticBeziers.push_back({ float64_t2(60.99808515344623, 86.57026372642981), float64_t2(61.33297230240677, 86.28756613643081), float64_t2(61.72470550254966, 86.17668431252241) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 61.72470550254966, 86.17668431252241 });
        linePoints.push_back({ 71.76994947575557, 83.33333333333334 });
        linePoints.push_back({ 69.39944453256167, 75 });
        linePoints.push_back({ 21.84019954342114, 75 });
        linePoints.push_back({ 22.885857597915642, 79.99322332648767 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(22.885857597915642, 79.99322332648767), float64_t2(23.13793416118432, 81.19693861722395), float64_t2(23.083474311036415, 82.44116667106196) });
        quadraticBeziers.push_back({ float64_t2(23.083474311036415, 82.44116667106196), float64_t2(23.029014460888508, 83.68539472834932), float64_t2(22.67347046362426, 84.8455692158529) });
        quadraticBeziers.push_back({ float64_t2(22.67347046362426, 84.8455692158529), float64_t2(22.31792646636001, 86.0057437033565), float64_t2(21.696893735804004, 86.9657133395473) });
        quadraticBeziers.push_back({ float64_t2(21.696893735804004, 86.9657133395473), float64_t2(21.07586100255217, 87.92568297573814), float64_t2(20.251514451842446, 88.58934005860377) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(20.251514451842446, 88.58934005860377), float64_t2(20.072180908124384, 88.73371620214095), float64_t2(19.937854675221015, 88.94337315664247) });
        quadraticBeziers.push_back({ float64_t2(19.937854675221015, 88.94337315664247), float64_t2(19.803528442317642, 89.15303011459334), float64_t2(19.72805937992723, 89.4063509621278) });
        quadraticBeziers.push_back({ float64_t2(19.72805937992723, 89.4063509621278), float64_t2(19.652590314840992, 89.65967180966227), float64_t2(19.643759744079773, 89.93053762242198) });
        quadraticBeziers.push_back({ float64_t2(19.643759744079773, 89.93053762242198), float64_t2(19.634929173318554, 90.20140343173234), float64_t2(19.693647578643837, 90.4618862774913) });
        quadraticBeziers.push_back({ float64_t2(19.693647578643837, 90.4618862774913), float64_t2(19.752365986664948, 90.7223691198009), float64_t2(19.87257914435553, 90.94561162163262) });
        quadraticBeziers.push_back({ float64_t2(19.87257914435553, 90.94561162163262), float64_t2(19.992792302046112, 91.16885412691367), float64_t2(20.16210549709703, 91.33183861289311) });
        quadraticBeziers.push_back({ float64_t2(20.16210549709703, 91.33183861289311), float64_t2(20.331418689452114, 91.49482310232189), float64_t2(20.532374692972475, 91.58074487759559) });
        quadraticBeziers.push_back({ float64_t2(20.532374692972475, 91.58074487759559), float64_t2(20.733330696492835, 91.6666666528693), float64_t2(20.94520971253289, 91.66666666666666) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 20.94520971253289, 91.66666666666666 });
        linePoints.push_back({ 30.931669882257395, 91.66666666666666 });
        linePoints.push_back({ 30.931669882257395, 100 });
        linePoints.push_back({ 18.774240110536073, 100 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(18.774240110536073, 100), float64_t2(17.632518402860846, 100), float64_t2(16.5508489004743, 99.53246279102233) });
        quadraticBeziers.push_back({ float64_t2(16.5508489004743, 99.53246279102233), float64_t2(15.469179400783585, 99.064925585494), float64_t2(14.561349269127163, 98.17903415403433) });
        quadraticBeziers.push_back({ float64_t2(14.561349269127163, 98.17903415403433), float64_t2(13.65351913747074, 97.29314272257464), float64_t2(13.015028361187145, 96.08208918944001) });
        quadraticBeziers.push_back({ float64_t2(13.015028361187145, 96.08208918944001), float64_t2(12.376537582207717, 94.87103565630538), float64_t2(12.07455275267436, 93.4622178544049) });
        quadraticBeziers.push_back({ float64_t2(12.07455275267436, 93.4622178544049), float64_t2(11.772567923141006, 92.05340005250441), float64_t2(11.838856595695546, 90.5950197984499) });
        quadraticBeziers.push_back({ float64_t2(11.838856595695546, 90.5950197984499), float64_t2(11.905145268250086, 89.13663954094604), float64_t2(12.332734151189161, 87.78211241587996) });
        quadraticBeziers.push_back({ float64_t2(12.332734151189161, 87.78211241587996), float64_t2(12.76032303682407, 86.42758528736455), float64_t2(13.50423154310238, 85.31940195157573) });
        quadraticBeziers.push_back({ float64_t2(13.50423154310238, 85.31940195157573), float64_t2(14.24814004938069, 84.2112186157869), float64_t2(15.230112080599223, 83.46595538228199) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(15.230112080599223, 83.46595538228199), float64_t2(15.535567719327014, 83.23413120168779), float64_t2(15.767366648704215, 82.88985185532106) });
        quadraticBeziers.push_back({ float64_t2(15.767366648704215, 82.88985185532106), float64_t2(15.999165578081415, 82.54557250895434), float64_t2(16.133141342214945, 82.12473121713157) });
        quadraticBeziers.push_back({ float64_t2(16.133141342214945, 82.12473121713157), float64_t2(16.267117109044307, 81.7038899253088), float64_t2(16.28930191912343, 81.2503619602433) });
        quadraticBeziers.push_back({ float64_t2(16.28930191912343, 81.2503619602433), float64_t2(16.311486726506725, 80.79683399862714), float64_t2(16.219567678436977, 80.35790242116761) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 16.219567678436977, 80.35790242116761 });
        linePoints.push_back({ 10.57862250293831, 53.421277252750265 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(10.57862250293831, 53.421277252750265), float64_t2(5.931895509130987, 55.65775133048495), float64_t2(0.9679965080696193, 55.550833598331174) });
        quadraticBeziers.push_back({ float64_t2(0.9679965080696193, 55.550833598331174), float64_t2(-3.995902492991749, 55.4439158661774), float64_t2(-8.581165148795344, 53.00859338776382) });
        quadraticBeziers.push_back({ float64_t2(-8.581165148795344, 53.00859338776382), float64_t2(-13.166427804598943, 50.57327090590088), float64_t2(-16.73032063516811, 46.15091197192669) });
        quadraticBeziers.push_back({ float64_t2(-16.73032063516811, 46.15091197192669), float64_t2(-20.294213465737272, 41.728553037952494), float64_t2(-22.337172238714288, 35.93905629866101) });
        quadraticBeziers.push_back({ float64_t2(-22.337172238714288, 35.93905629866101), float64_t2(-24.380131011691308, 30.149559559369525), float64_t2(-24.61578652632701, 23.804460507300167) });
        quadraticBeziers.push_back({ float64_t2(-24.61578652632701, 23.804460507300167), float64_t2(-24.674045424619525, 22.235821287527127), float64_t2(-24.616785757529833, 20.667121835328913) });
        quadraticBeziers.push_back({ float64_t2(-24.616785757529833, 20.667121835328913), float64_t2(-24.55952609044014, 19.098422379681356), float64_t2(-24.387255102770435, 17.54355918340109) });
        quadraticBeziers.push_back({ float64_t2(-24.387255102770435, 17.54355918340109), float64_t2(-24.041946689293663, 14.426915188906369), float64_t2(-23.24676157510896, 11.448076436365092) });
        quadraticBeziers.push_back({ float64_t2(-23.24676157510896, 11.448076436365092), float64_t2(-21.642081114646867, 5.436791417499384), float64_t2(-18.421998759245177, 0.6019448206104614) });
        quadraticBeziers.push_back({ float64_t2(-18.421998759245177, 0.6019448206104614), float64_t2(-15.201916403843482, -4.232901779727803), float64_t2(-10.817803185497988, -7.213591359969643) });
        quadraticBeziers.push_back({ float64_t2(-10.817803185497988, -7.213591359969643), float64_t2(-6.433689967152493, -10.194280943660825), float64_t2(-1.5000834922884798, -10.902999062091112) });
        quadraticBeziers.push_back({ float64_t2(-1.5000834922884798, -10.902999062091112), float64_t2(3.4335229825755347, -11.611717183970743), float64_t2(8.225060575110845, -9.949120161709962) });
        quadraticBeziers.push_back({ float64_t2(8.225060575110845, -9.949120161709962), float64_t2(13.016598167646157, -8.286523139449182), float64_t2(16.99441907197321, -4.485663443941761) });
        quadraticBeziers.push_back({ float64_t2(16.99441907197321, -4.485663443941761), float64_t2(20.972239973604427, -0.6848037449849976), float64_t2(23.578758098071752, 4.721537860179389) });
        quadraticBeziers.push_back({ float64_t2(23.578758098071752, 4.721537860179389), float64_t2(26.1852762252349, 10.127879461894434), float64_t2(27.055126143864243, 16.381875777410137) });
        quadraticBeziers.push_back({ float64_t2(27.055126143864243, 16.381875777410137), float64_t2(27.48617252398593, 19.480987952125293), float64_t2(27.45618282044724, 22.628559596422644) });
        quadraticBeziers.push_back({ float64_t2(27.45618282044724, 22.628559596422644), float64_t2(27.426193116908543, 25.776131244169342), float64_t2(26.93622764573934, 28.860876974822197) });
        quadraticBeziers.push_back({ float64_t2(26.93622764573934, 28.860876974822197), float64_t2(25.947479228985088, 35.08588186361724), float64_t2(23.238729056491316, 40.40931302157265) });
        quadraticBeziers.push_back({ float64_t2(23.238729056491316, 40.40931302157265), float64_t2(20.529978881301716, 45.73274417952807), float64_t2(16.480922627682048, 49.40839628516524) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 16.480922627682048, 49.40839628516524 });
        linePoints.push_back({ 17.325954543192257, 53.44359025842061 });
        linePoints.push_back({ 47.50165856850336, 26.40849214254154 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(47.50165856850336, 26.40849214254154), float64_t2(47.05459163774403, 24.5989253261575), float64_t2(47.00332962506486, 22.702232279159404) });
        quadraticBeziers.push_back({ float64_t2(47.00332962506486, 22.702232279159404), float64_t2(46.952067609689855, 20.805539228711968), float64_t2(47.30036888315071, 18.960779570732957) });
        quadraticBeziers.push_back({ float64_t2(47.30036888315071, 18.960779570732957), float64_t2(47.6486701593074, 17.116019912753945), float64_t2(48.37099835509587, 15.458445678706523) });
        quadraticBeziers.push_back({ float64_t2(48.37099835509587, 15.458445678706523), float64_t2(49.093326550884335, 13.800871444659101), float64_t2(50.13672281962545, 12.452010813824556) });
        quadraticBeziers.push_back({ float64_t2(50.13672281962545, 12.452010813824556), float64_t2(52.263220149623926, 9.70296078465051), float64_t2(55.19415626979962, 8.756009451355096) });
        quadraticBeziers.push_back({ float64_t2(55.19415626979962, 8.756009451355096), float64_t2(58.12509239267114, 7.80905811805968), float64_t2(61.01667373080921, 8.936825349788975) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 61.01667373080921, 8.936825349788975 });
        linePoints.push_back({ 62.65388740834709, 4.444444445134313 });
        linePoints.push_back({ 59.713115148309456, 4.444444445134313 });
        linePoints.push_back({ 59.713115148309456, 0 });
        linePoints.push_back({ 62.64628497716216, 0 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(62.64628497716216, 0), float64_t2(63.072477023529395, -0.7831520262967658), float64_t2(63.73359004885505, -1.2249093502759933) });
        quadraticBeziers.push_back({ float64_t2(63.73359004885505, -1.2249093502759933), float64_t2(64.39470307418071, -1.6666666708058782), float64_t2(65.14053915599732, -1.6666666708058782) });
        quadraticBeziers.push_back({ float64_t2(65.14053915599732, -1.6666666708058782), float64_t2(65.88637523781392, -1.6666666708058782), float64_t2(66.54748826313958, -1.2249093502759933) });
        quadraticBeziers.push_back({ float64_t2(66.54748826313958, -1.2249093502759933), float64_t2(67.20860128846525, -0.7831520262967658), float64_t2(67.63479333213664, 0) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 67.63479333213664, 0 });
        linePoints.push_back({ 70.56796315829352, 0 });
        linePoints.push_back({ 70.56796315829352, 4.444444445134313 });
        linePoints.push_back({ 66.00744729793786, 4.444444445134313 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    polylines.push_back(polyline);
}
{
    CPolyline polyline;
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(0, 25.02802262358643), float64_t2(2.717404740558329, 27.27083834922976), float64_t2(4.617262276332434, 30.61899776329045) });
        quadraticBeziers.push_back({ float64_t2(4.617262276332434, 30.61899776329045), float64_t2(6.517119812106538, 33.96715717735114), float64_t2(7.342404549629493, 37.96769858993314) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 7.342404549629493, 37.96769858993314 });
        linePoints.push_back({ 8.541846269830518, 43.695269231856976 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(8.541846269830518, 43.695269231856976), float64_t2(5.410242206424399, 45.398905031658984), float64_t2(2.009195213831284, 45.54280823579541) });
        quadraticBeziers.push_back({ float64_t2(2.009195213831284, 45.54280823579541), float64_t2(-1.3918517787618319, 45.686711439931834), float64_t2(-4.6045778431016835, 44.25151340180525) });
        quadraticBeziers.push_back({ float64_t2(-4.6045778431016835, 44.25151340180525), float64_t2(-7.817303907441535, 42.81631536367867), float64_t2(-10.409292237887618, 39.995186537918116) });
        quadraticBeziers.push_back({ float64_t2(-10.409292237887618, 39.995186537918116), float64_t2(-13.001280568333701, 37.174057708708226), float64_t2(-14.62366254577411, 33.34670787401222) });
        quadraticBeziers.push_back({ float64_t2(-14.62366254577411, 33.34670787401222), float64_t2(-16.24604452321452, 29.519358039316202), float64_t2(-16.68045567472724, 25.200929271954074) });
        quadraticBeziers.push_back({ float64_t2(-16.68045567472724, 25.200929271954074), float64_t2(-16.895802672035362, 23.060190781123108), float64_t2(-16.800590354091316, 20.905233357377625) });
        quadraticBeziers.push_back({ float64_t2(-16.800590354091316, 20.905233357377625), float64_t2(-16.70537803614727, 18.750275930182802), float64_t2(-16.302837590540957, 16.654231654549086) });
        quadraticBeziers.push_back({ float64_t2(-16.302837590540957, 16.654231654549086), float64_t2(-15.490808354841956, 12.42596280105688), float64_t2(-13.541633794297244, 8.856957623114187) });
        quadraticBeziers.push_back({ float64_t2(-13.541633794297244, 8.856957623114187), float64_t2(-11.592459233752534, 5.287952448620841), float64_t2(-8.768488457952525, 2.85858113180708) });
        quadraticBeziers.push_back({ float64_t2(-8.768488457952525, 2.85858113180708), float64_t2(-5.944517684848347, 0.42920981844266254), float64_t2(-2.625842989797497, -0.5335464487197222) });
        quadraticBeziers.push_back({ float64_t2(-2.625842989797497, -0.5335464487197222), float64_t2(0.6928317052533538, -1.4963027124327641), float64_t2(4.059533359572301, -0.8628617661694685) });
        quadraticBeziers.push_back({ float64_t2(4.059533359572301, -0.8628617661694685), float64_t2(7.426235016587078, -0.2294208233555158), float64_t2(10.387822486109199, 1.914959362949486) });
        quadraticBeziers.push_back({ float64_t2(10.387822486109199, 1.914959362949486), float64_t2(13.349409955631323, 4.0593395458051456), float64_t2(15.507268431015476, 7.426036156162068) });
        quadraticBeziers.push_back({ float64_t2(15.507268431015476, 7.426036156162068), float64_t2(17.66512690639963, 10.792732766518991), float64_t2(18.728819464180425, 14.928605338489568) });
        quadraticBeziers.push_back({ float64_t2(18.728819464180425, 14.928605338489568), float64_t2(19.256114860103565, 16.97884678012795), float64_t2(19.480531731669423, 19.11805957486784) });
        quadraticBeziers.push_back({ float64_t2(19.480531731669423, 19.11805957486784), float64_t2(19.592501152409717, 20.18538761401066), float64_t2(19.62715963567816, 21.261374483367913) });
        quadraticBeziers.push_back({ float64_t2(19.62715963567816, 21.261374483367913), float64_t2(19.661818116250775, 22.33736135272516), float64_t2(19.618870986188988, 23.41285874998128) });
        quadraticBeziers.push_back({ float64_t2(19.618870986188988, 23.41285874998128), float64_t2(19.445229947720925, 27.76123959295176), float64_t2(18.057626525809063, 31.736858461604072) });
        quadraticBeziers.push_back({ float64_t2(18.057626525809063, 31.736858461604072), float64_t2(16.6700231038972, 35.712477333705735), float64_t2(14.255221756610096, 38.78023588891934) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 14.255221756610096, 38.78023588891934 });
        linePoints.push_back({ 12.789863693152977, 31.78286227846035 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(12.789863693152977, 31.78286227846035), float64_t2(12.273817284201943, 29.34820903810086), float64_t2(11.248978207963201, 27.193208349247772) });
        quadraticBeziers.push_back({ float64_t2(11.248978207963201, 27.193208349247772), float64_t2(10.224139134420291, 25.038207656945342), float64_t2(8.77135628800617, 23.332866129499894) });
        quadraticBeziers.push_back({ float64_t2(8.77135628800617, 23.332866129499894), float64_t2(5.806132868094706, 19.852154156951993), float64_t2(1.846214120639083, 18.936716589248842) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(1.846214120639083, 18.936716589248842), float64_t2(1.4019829584192953, 18.839378855018705), float64_t2(0.9580133564739998, 18.938651056615292) });
        quadraticBeziers.push_back({ float64_t2(0.9580133564739998, 18.938651056615292), float64_t2(0.5140437518328735, 19.03792325821188), float64_t2(0.12194248992922319, 19.32226605023499) });
        quadraticBeziers.push_back({ float64_t2(0.12194248992922319, 19.32226605023499), float64_t2(-0.2701587692785965, 19.606608838808757), float64_t2(-0.5648140476944825, 20.042970369535464) });
        quadraticBeziers.push_back({ float64_t2(-0.5648140476944825, 20.042970369535464), float64_t2(-0.8594693288061993, 20.479331896812827), float64_t2(-1.0224280600830296, 21.016989738025046) });
        quadraticBeziers.push_back({ float64_t2(-1.0224280600830296, 21.016989738025046), float64_t2(-1.185386788664029, 21.554647575787925), float64_t2(-1.1977067348054775, 22.13110467074094) });
        quadraticBeziers.push_back({ float64_t2(-1.1977067348054775, 22.13110467074094), float64_t2(-1.210026680946926, 22.7075617691433), float64_t2(-1.0702757789618365, 23.255811062537962) });
        quadraticBeziers.push_back({ float64_t2(-1.0702757789618365, 23.255811062537962), float64_t2(-0.9305248769767469, 23.804060355932624), float64_t2(-0.6549476956394068, 24.260373644668746) });
        quadraticBeziers.push_back({ float64_t2(-0.6549476956394068, 24.260373644668746), float64_t2(-0.37937051699789726, 24.716686936854213), float64_t2(0, 25.02802262358643) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    polylines.push_back(polyline);
}
{
    CPolyline polyline;
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 48.068592563516596, 44.44444444444444 });
        linePoints.push_back({ 45.13542273466389, 44.44444444444444 });
        linePoints.push_back({ 45.13542273466389, 39.999999999310134 });
        linePoints.push_back({ 49.69593859501955, 39.999999999310134 });
        linePoints.push_back({ 51.93211884462939, 33.86410363018513 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(51.93211884462939, 33.86410363018513), float64_t2(51.575063246586026, 33.56687499003278), float64_t2(51.2373095912247, 33.234759178702475) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 51.2373095912247, 33.234759178702475 });
        linePoints.push_back({ 19.119801595116574, 62.00956354193665 });
        linePoints.push_back({ 20.095070888197338, 66.66666666666666 });
        linePoints.push_back({ 67.02893958936777, 66.66666666666666 });
        linePoints.push_back({ 58.3331992002567, 36.097439878654704 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(58.3331992002567, 36.097439878654704), float64_t2(56.47036206743867, 36.2032737100014), float64_t2(54.6867121621482, 35.50761909465547) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    {
        std::vector<float64_t2> linePoints;
        linePoints.push_back({ 54.6867121621482, 35.50761909465547 });
        linePoints.push_back({ 53.049498484610325, 39.999999999310134 });
        linePoints.push_back({ 55.99027074464795, 39.999999999310134 });
        linePoints.push_back({ 55.99027074464795, 44.44444444444444 });
        linePoints.push_back({ 53.05710091579525, 44.44444444444444 });
        polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
    }
    {
        std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
        quadraticBeziers.push_back({ float64_t2(53.05710091579525, 44.44444444444444), float64_t2(52.630908869428026, 45.22759647074121), float64_t2(51.969795844102364, 45.669353787821755) });
        quadraticBeziers.push_back({ float64_t2(51.969795844102364, 45.669353787821755), float64_t2(51.30868282147253, 46.111111108351636), float64_t2(50.56284673965592, 46.111111108351636) });
        quadraticBeziers.push_back({ float64_t2(50.56284673965592, 46.111111108351636), float64_t2(49.817010660535146, 46.111111108351636), float64_t2(49.15589763520949, 45.66935379127109) });
        quadraticBeziers.push_back({ float64_t2(49.15589763520949, 45.66935379127109), float64_t2(48.49478460988382, 45.22759647074121), float64_t2(48.068592563516596, 44.44444444444444) });
        polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
    }
    polylines.push_back(polyline);
}
