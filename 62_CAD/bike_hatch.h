std::vector<CPolyline> polylines;
{
CPolyline polyline;
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(97.93187369058725, 19.1666666659768), float64_t2(98.39675081804235, 19.1666666659768), float64_t2(98.8276811965705, 19.38979268349983) } );
    quadraticBeziers.push_back( { float64_t2(98.8276811965705, 19.38979268349983), float64_t2(99.2587787866915, 19.613005272630186), float64_t2(99.5929092483342, 20.026893354952335) } );
    quadraticBeziers.push_back( { float64_t2(99.5929092483342, 20.026893354952335), float64_t2(99.9338921637665, 20.449269590554415), float64_t2(100.12274019431064, 21.00637382655232) } );
    quadraticBeziers.push_back( { float64_t2(100.12274019431064, 21.00637382655232), float64_t2(100.21821729135098, 21.288032550364733), float64_t2(100.26799228922671, 21.588364943723988) } );
    quadraticBeziers.push_back( { float64_t2(100.26799228922671, 21.588364943723988), float64_t2(100.31994025601874, 21.90180857907291), float64_t2(100.31994025601874, 22.22222222222222) } );
    quadraticBeziers.push_back( { float64_t2(100, 23.750000002069605), float64_t2(100.15383678435397, 23.409070509175457), float64_t2(100.23511234519407, 23.02938287525817) } );
    quadraticBeziers.push_back( { float64_t2(100.23511234519407, 23.02938287525817), float64_t2(100.31994025601874, 22.633100063022642), float64_t2(100.31994025601874, 22.22222222222222) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(78.85003878081946, 19.1666666659768), float64_t2(79.2966051820908, 14.840927295800713), float64_t2(80.93467945504369, 11.013974301103088) } );
    quadraticBeziers.push_back( { float64_t2(80.93467945504369, 11.013974301103088), float64_t2(82.57234077526357, 7.187986063460509), float64_t2(85.17968162681255, 4.377432039904374) } );
    quadraticBeziers.push_back( { float64_t2(85.17968162681255, 4.377432039904374), float64_t2(87.78625320421031, 1.5677072487219617), float64_t2(91.00999702873442, 0.1522232054008378) } );
    quadraticBeziers.push_back( { float64_t2(91.00999702873442, 0.1522232054008378), float64_t2(94.23262415123446, -1.2627705138314653), float64_t2(97.63715261772813, -1.0935189443881865) } );
    quadraticBeziers.push_back( { float64_t2(97.63715261772813, -1.0935189443881865), float64_t2(101.0961087758057, -0.9215615724247915), float64_t2(104.26522650369643, 0.8601912124841302) } );
    quadraticBeziers.push_back( { float64_t2(104.26522650369643, 0.8601912124841302), float64_t2(107.43367041679981, 2.641565162964441), float64_t2(109.87175858391298, 5.7843377399775715) } );
    quadraticBeziers.push_back( { float64_t2(109.87175858391298, 5.7843377399775715), float64_t2(112.35968809868622, 8.991357384042606), float64_t2(113.73205635439572, 13.154911153294421) } );
    quadraticBeziers.push_back( { float64_t2(113.73205635439572, 13.154911153294421), float64_t2(115.16529044404706, 17.50312248520829), float64_t2(115.16529044404706, 22.22222222222222) } );
    quadraticBeziers.push_back( { float64_t2(90.77780091723362, 44.18802270665765), float64_t2(94.87779262947645, 46.06774431411867), float64_t2(99.19916490998988, 45.37407421610421) } );
    quadraticBeziers.push_back( { float64_t2(99.19916490998988, 45.37407421610421), float64_t2(103.52231153502657, 44.68011929895039), float64_t2(107.13337555768155, 41.560640986318944) } );
    quadraticBeziers.push_back( { float64_t2(107.13337555768155, 41.560640986318944), float64_t2(110.81732121141246, 38.37820265304159), float64_t2(112.94357180011761, 33.383712683010984) } );
    quadraticBeziers.push_back( { float64_t2(112.94357180011761, 33.383712683010984), float64_t2(114.01378030462519, 30.869829478777117), float64_t2(114.57731607568466, 28.099487363188352) } );
    quadraticBeziers.push_back( { float64_t2(114.57731607568466, 28.099487363188352), float64_t2(115.16529044404706, 25.209005094236797), float64_t2(115.16529044404706, 22.22222222222222) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 97.93187369058725, 19.1666666659768 });
    linePoints.push_back({ 78.85003878081946, 19.1666666659768 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
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
    quadraticBeziers.push_back( { float64_t2(78.85003878081946, 25.277777778467648), float64_t2(79.29464073567648, 29.58448823127482), float64_t2(80.92077276993082, 33.39791925003131) } );
    quadraticBeziers.push_back( { float64_t2(80.92077276993082, 33.39791925003131), float64_t2(82.54716290840473, 37.211955542227734), float64_t2(85.13745643750717, 40.02135603999098) } );
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
    quadraticBeziers.push_back( { float64_t2(50.904590224402234, 22.222222225671565), float64_t2(50.904590224402234, 23.154334641165203), float64_t2(51.055711576705896, 24.066170674093343) } );
    quadraticBeziers.push_back( { float64_t2(51.055711576705896, 24.066170674093343), float64_t2(51.20051157542771, 24.93986491182888), float64_t2(51.47826312751328, 25.759235741915525) } );
    quadraticBeziers.push_back( { float64_t2(51.47826312751328, 25.759235741915525), float64_t2(52.02763921514359, 27.379902607450884), float64_t2(53.0195895158084, 28.608633474343353) } );
    quadraticBeziers.push_back( { float64_t2(53.0195895158084, 28.608633474343353), float64_t2(53.99160539687126, 29.81267153388924), float64_t2(55.24570747771344, 30.462017255248846) } );
    quadraticBeziers.push_back( { float64_t2(55.24570747771344, 30.462017255248846), float64_t2(56.499323120174495, 31.111111119389534), float64_t2(57.85169294647871, 31.111111119389534) } );
    quadraticBeziers.push_back( { float64_t2(50.904590224402234, 22.222222225671565), float64_t2(50.904590224402234, 21.290109810177928), float64_t2(51.055711576705896, 20.378273777249785) } );
    quadraticBeziers.push_back( { float64_t2(51.055711576705896, 20.378273777249785), float64_t2(51.20051157542771, 19.50457953951425), float64_t2(51.47826312751328, 18.685208709427602) } );
    quadraticBeziers.push_back( { float64_t2(51.47826312751328, 18.685208709427602), float64_t2(52.02763921514359, 17.064541843892247), float64_t2(53.0195895158084, 15.835810970101091) } );
    quadraticBeziers.push_back( { float64_t2(53.0195895158084, 15.835810970101091), float64_t2(53.99160539687126, 14.631772917453889), float64_t2(55.24570747771344, 13.982427196094282) } );
    quadraticBeziers.push_back( { float64_t2(55.24570747771344, 13.982427196094282), float64_t2(56.499323120174495, 13.333333338852283), float64_t2(57.85169294647871, 13.333333331953595) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(57.85169294647871, 13.333333331953595), float64_t2(59.204062772782926, 13.333333331953595), float64_t2(60.45767842063564, 13.982427196094282) } );
    quadraticBeziers.push_back( { float64_t2(60.45767842063564, 13.982427196094282), float64_t2(61.71178050147782, 14.631772910555204), float64_t2(62.6837963852365, 15.835810970101091) } );
    quadraticBeziers.push_back( { float64_t2(62.6837963852365, 15.835810970101091), float64_t2(63.67574668859716, 17.064541836993563), float64_t2(64.22512277622747, 18.685208705978262) } );
    quadraticBeziers.push_back( { float64_t2(64.22512277622747, 18.685208705978262), float64_t2(64.50287432831303, 19.504579536064906), float64_t2(64.64767432703485, 20.378273773800444) } );
    quadraticBeziers.push_back( { float64_t2(64.64767432703485, 20.378273773800444), float64_t2(64.79879567933851, 21.290109810177928), float64_t2(64.79879567933851, 22.222222225671565) } );
    quadraticBeziers.push_back( { float64_t2(57.85169294647871, 31.111111119389534), float64_t2(59.204062772782926, 31.111111119389534), float64_t2(60.45767842063564, 30.462017265596874) } );
    quadraticBeziers.push_back( { float64_t2(60.45767842063564, 30.462017265596874), float64_t2(61.71178050147782, 29.812671540787928), float64_t2(62.6837963852365, 28.60863348124204) } );
    quadraticBeziers.push_back( { float64_t2(62.6837963852365, 28.60863348124204), float64_t2(63.67574668859716, 27.379902614349565), float64_t2(64.22512277622747, 25.759235745364865) } );
    quadraticBeziers.push_back( { float64_t2(64.22512277622747, 25.759235745364865), float64_t2(64.50287432831303, 24.939864915278225), float64_t2(64.64767432703485, 24.066170677542686) } );
    quadraticBeziers.push_back( { float64_t2(64.64767432703485, 24.066170677542686), float64_t2(64.79879567933851, 23.154334641165203), float64_t2(64.79879567933851, 22.222222225671565) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 57.85169294647871, 31.111111119389534 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
polylines.push_back(polyline);
}
{
CPolyline polyline;
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(70.98719451596574, 25.277777778467648), float64_t2(71.49003945562686, 32.267083738137174), float64_t2(74.15294587114202, 38.40346986565877) } );
    quadraticBeziers.push_back( { float64_t2(74.15294587114202, 38.40346986565877), float64_t2(76.8154453914557, 44.538918344510925), float64_t2(81.18532078502435, 48.779986981578446) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 64.453331546595, 33.24727201834321 });
    linePoints.push_back({ 73.6329219204022, 65.5174373511087 });
    linePoints.push_back({ 81.18532078502435, 48.779986981578446 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(64.453331546595, 33.24727201834321), float64_t2(65.96643046620333, 31.76398218298952), float64_t2(66.99404435825028, 29.70983203224562) } );
    quadraticBeziers.push_back( { float64_t2(66.99404435825028, 29.70983203224562), float64_t2(68.02136512184562, 27.65626782827355), float64_t2(68.4405957893852, 25.277777778467648) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 70.98719450787826, 25.277777778467648 });
    linePoints.push_back({ 68.4405957893852, 25.277777778467648 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 64.453331546595, 33.24727201834321 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
polylines.push_back(polyline);
}
{
CPolyline polyline;
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(63.77126705102386, 10.580340814259317), float64_t2(65.54957590381235, 12.060682585945836), float64_t2(66.76141580046492, 14.288787971492168) } );
    quadraticBeziers.push_back( { float64_t2(66.76141580046492, 14.288787971492168), float64_t2(67.97368956410477, 16.51769107276643), float64_t2(68.44059578668937, 19.1666666659768) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 66.00744729793786, 4.444444445134313 });
    linePoints.push_back({ 63.77126705102386, 10.580340814259317 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(70.98719450787826, 19.1666666659768), float64_t2(71.44077246277757, 12.862148411847926), float64_t2(73.66932352777896, 7.209452038147936) } );
    quadraticBeziers.push_back( { float64_t2(73.66932352777896, 7.209452038147936), float64_t2(75.89761135569918, 1.5574233675444569), float64_t2(79.59017059817272, -2.655866990486781) } );
    quadraticBeziers.push_back( { float64_t2(79.59017059817272, -2.655866990486781), float64_t2(83.27885002541488, -6.864730392893155), float64_t2(87.91597264531426, -9.052588331892535) } );
    quadraticBeziers.push_back( { float64_t2(87.91597264531426, -9.052588331892535), float64_t2(92.55186302527888, -11.239864887600696), float64_t2(97.49173762185244, -11.103337613382825) } );
    quadraticBeziers.push_back( { float64_t2(97.49173762185244, -11.103337613382825), float64_t2(102.49247476615024, -10.965128232621485), float64_t2(107.0976234028704, -8.467084576410276) } );
    quadraticBeziers.push_back( { float64_t2(107.0976234028704, -8.467084576410276), float64_t2(111.70117115289979, -5.969909312962382), float64_t2(115.25503701923337, -1.4694359912364572) } );
    quadraticBeziers.push_back( { float64_t2(115.25503701923337, -1.4694359912364572), float64_t2(118.88251623248127, 3.1242583257456618), float64_t2(120.8871345593627, 9.129733064522345) } );
    quadraticBeziers.push_back( { float64_t2(120.8871345593627, 9.129733064522345), float64_t2(121.90119876167265, 12.167686399900251), float64_t2(122.42951203585932, 15.401218372776551) } );
    quadraticBeziers.push_back( { float64_t2(122.42951203585932, 15.401218372776551), float64_t2(122.98078100099144, 18.77525025219829), float64_t2(122.98078100099144, 22.22222222222222) } );
    quadraticBeziers.push_back( { float64_t2(86.82566526205497, 52.946653644795774), float64_t2(92.72969306110149, 56.125159830682804), float64_t2(99.11251401376924, 55.438282727091405) } );
    quadraticBeziers.push_back( { float64_t2(99.11251401376924, 55.438282727091405), float64_t2(105.497864407738, 54.75113342205683), float64_t2(110.90553726648005, 50.35248570213163) } );
    quadraticBeziers.push_back( { float64_t2(110.90553726648005, 50.35248570213163), float64_t2(116.42290456411789, 45.86461158124385), float64_t2(119.63031359311115, 38.57517756728662) } );
    quadraticBeziers.push_back( { float64_t2(119.63031359311115, 38.57517756728662), float64_t2(121.24270324504418, 34.91072196306454), float64_t2(122.09326705090918, 30.848692823201418) } );
    quadraticBeziers.push_back( { float64_t2(122.09326705090918, 30.848692823201418), float64_t2(122.98078100099144, 26.61020124883012), float64_t2(122.98078100099144, 22.22222222222222) } );
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
    quadraticBeziers.push_back( { float64_t2(78.54280859807695, 82.77777777984738), float64_t2(79.7613882277515, 82.90389291252251), float64_t2(80.89165578495763, 83.50013184050718) } );
    quadraticBeziers.push_back( { float64_t2(80.89165578495763, 83.50013184050718), float64_t2(82.02202446277205, 84.09642411257934), float64_t2(82.95605542119166, 85.1059447284098) } );
    quadraticBeziers.push_back( { float64_t2(82.95605542119166, 85.1059447284098), float64_t2(83.88978589154198, 86.1151405679131), float64_t2(84.53811878703992, 87.44091421227765) } );
    quadraticBeziers.push_back( { float64_t2(84.53811878703992, 87.44091421227765), float64_t2(85.18632360092711, 88.76642594803815), float64_t2(85.48735680493138, 90.28184683993459) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 86.82566526205497, 52.946653644795774 });
    linePoints.push_back({ 76.54077671044429, 75.73978399374971 });
    linePoints.push_back({ 78.54280859807695, 82.77777777984738 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(85.48735681032304, 90.28184683993459), float64_t2(85.51411703003681, 90.41655954594412), float64_t2(85.51411703003681, 90.55555555279608) } );
    quadraticBeziers.push_back( { float64_t2(84.64572918977726, 91.66666666666666), float64_t2(84.81477541671737, 91.66666666666666), float64_t2(84.97147737303594, 91.58552993019974) } );
    quadraticBeziers.push_back( { float64_t2(84.97147737303594, 91.58552993019974), float64_t2(85.12824013381517, 91.50436171502979), float64_t2(85.24974211962198, 91.35385695844889) } );
    quadraticBeziers.push_back( { float64_t2(85.24974211962198, 91.35385695844889), float64_t2(85.37373590754206, 91.20026559879383), float64_t2(85.44240791815886, 90.99768224275775) } );
    quadraticBeziers.push_back( { float64_t2(85.44240791815886, 90.99768224275775), float64_t2(85.47712686351748, 90.89526088770342), float64_t2(85.4952268620098, 90.78604910798647) } );
    quadraticBeziers.push_back( { float64_t2(85.4952268620098, 90.78604910798647), float64_t2(85.51411703003681, 90.67206960516395), float64_t2(85.51411703003681, 90.55555555279608) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(60.02266254847552, 88.88888888888889), float64_t2(60.02266254847552, 89.1801740180839), float64_t2(60.06988797258682, 89.4651227803142) } );
    quadraticBeziers.push_back( { float64_t2(60.06988797258682, 89.4651227803142), float64_t2(60.11513797016551, 89.73815222788188), float64_t2(60.201935330866206, 89.99420561379304) } );
    quadraticBeziers.push_back( { float64_t2(60.201935330866206, 89.99420561379304), float64_t2(60.37361535740824, 90.50066401078193), float64_t2(60.68359982720843, 90.88464240647025) } );
    quadraticBeziers.push_back( { float64_t2(60.68359982720843, 90.88464240647025), float64_t2(60.98735479037756, 91.2609042979225), float64_t2(61.379261690977714, 91.46382483757205) } );
    quadraticBeziers.push_back( { float64_t2(61.379261690977714, 91.46382483757205), float64_t2(61.77101657907831, 91.66666666666666), float64_t2(62.19363215047233, 91.66666666666666) } );
    quadraticBeziers.push_back( { float64_t2(60.02266254847552, 88.88888888888889), float64_t2(60.02266254847552, 88.3939976528011), float64_t2(60.156241324585245, 87.92955684993002) } );
    quadraticBeziers.push_back( { float64_t2(60.156241324585245, 87.92955684993002), float64_t2(60.28414988726875, 87.48483083176392), float64_t2(60.52006133058868, 87.11952022449286) } );
    quadraticBeziers.push_back( { float64_t2(60.52006133058868, 87.11952022449286), float64_t2(60.75126538179624, 86.76149904796922), float64_t2(61.06091597836018, 86.51918102124775) } );
    quadraticBeziers.push_back( { float64_t2(61.06091597836018, 86.51918102124775), float64_t2(61.37044261254262, 86.27696000039577), float64_t2(61.72470550254966, 86.17668431252241) } );
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
    quadraticBeziers.push_back( { float64_t2(22.885857597915642, 79.99322332648767), float64_t2(23.092984783770635, 80.98229647234633), float64_t2(23.092984783770635, 82.00626033875677) } );
    quadraticBeziers.push_back( { float64_t2(20.251514451842446, 88.58934005860377), float64_t2(21.54903699939049, 87.54474294406397), float64_t2(22.304227028621217, 85.83777460304124) } );
    quadraticBeziers.push_back( { float64_t2(22.304227028621217, 85.83777460304124), float64_t2(22.683781236824828, 84.97986203680435), float64_t2(22.884048709018227, 84.0279153789635) } );
    quadraticBeziers.push_back( { float64_t2(22.884048709018227, 84.0279153789635), float64_t2(23.092984783770635, 83.03476358867354), float64_t2(23.092984783770635, 82.00626033875677) } );
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
    quadraticBeziers.push_back( { float64_t2(19.642627956187294, 89.99999999241145), float64_t2(19.642627956187294, 90.17477106923857), float64_t2(19.670963209575742, 90.34574032519703) } );
    quadraticBeziers.push_back( { float64_t2(19.670963209575742, 90.34574032519703), float64_t2(19.698113208662125, 90.50955799304776), float64_t2(19.750191624004213, 90.66319002390459) } );
    quadraticBeziers.push_back( { float64_t2(19.750191624004213, 90.66319002390459), float64_t2(19.853199639929425, 90.96706506140806), float64_t2(20.039190321809546, 91.19745209744131) } );
    quadraticBeziers.push_back( { float64_t2(20.039190321809546, 91.19745209744131), float64_t2(20.221443299171852, 91.42320923231266), float64_t2(20.45658743899278, 91.54496155679226) } );
    quadraticBeziers.push_back( { float64_t2(20.45658743899278, 91.54496155679226), float64_t2(20.691640370774806, 91.66666666666666), float64_t2(20.94520971253289, 91.66666666666666) } );
    quadraticBeziers.push_back( { float64_t2(19.642627956187294, 89.99999999241145), float64_t2(19.642627956187294, 89.77960644082891), float64_t2(19.687399972205668, 89.56678820153078) } );
    quadraticBeziers.push_back( { float64_t2(19.687399972205668, 89.56678820153078), float64_t2(19.73031442879979, 89.36279963150069), float64_t2(19.811647472837173, 89.17896122568183) } );
    quadraticBeziers.push_back( { float64_t2(19.811647472837173, 89.17896122568183), float64_t2(19.973473906709533, 88.81318229768011), float64_t2(20.251514451842446, 88.58934005860377) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(11.827137385763772, 91.11111110973137), float64_t2(11.827137385763772, 92.04322352522502), float64_t2(11.978258738067437, 92.9550595616025) } );
    quadraticBeziers.push_back( { float64_t2(11.978258738067437, 92.9550595616025), float64_t2(12.123058736789245, 93.82875379588869), float64_t2(12.400810288874817, 94.64812462597534) } );
    quadraticBeziers.push_back( { float64_t2(12.400810288874817, 94.64812462597534), float64_t2(12.95018637650512, 96.26879149496004), float64_t2(13.94213667716994, 97.4975223618525) } );
    quadraticBeziers.push_back( { float64_t2(13.94213667716994, 97.4975223618525), float64_t2(14.914152560928628, 98.7015604213984), float64_t2(16.1682546417708, 99.350906142758) } );
    quadraticBeziers.push_back( { float64_t2(16.1682546417708, 99.350906142758), float64_t2(17.421870284231858, 100), float64_t2(18.774240110536073, 100) } );
    quadraticBeziers.push_back( { float64_t2(11.827137385763772, 91.11111110973137), float64_t2(11.827137385763772, 89.90462657616094), float64_t2(12.078473150940548, 88.74179084474841) } );
    quadraticBeziers.push_back( { float64_t2(12.078473150940548, 88.74179084474841), float64_t2(12.319362033122282, 87.62728889576263), float64_t2(12.775085171011217, 86.62873713506592) } );
    quadraticBeziers.push_back( { float64_t2(12.775085171011217, 86.62873713506592), float64_t2(13.682471008207314, 84.64053057617059), float64_t2(15.230112077903394, 83.46595538228199) } );
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
    quadraticBeziers.push_back( { float64_t2(16.21956768113281, 80.35790242116761), float64_t2(16.29354167357775, 80.7111428257216), float64_t2(16.29354167357775, 81.07684420559693) } );
    quadraticBeziers.push_back( { float64_t2(15.230112080599223, 83.46595538228199), float64_t2(15.71374991690112, 83.09890063203595), float64_t2(15.99730799169886, 82.47758607828507) } );
    quadraticBeziers.push_back( { float64_t2(15.99730799169886, 82.47758607828507), float64_t2(16.139721471278214, 82.16553865069592), float64_t2(16.214999248307922, 81.81725679034436) } );
    quadraticBeziers.push_back( { float64_t2(16.214999248307922, 81.81725679034436), float64_t2(16.29354167357775, 81.45387062320003), float64_t2(16.29354167357775, 81.07684420559693) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(-24.645151937487675, 22.22222222222222), float64_t2(-24.645151937487675, 26.523936484698897), float64_t2(-23.791624823334363, 30.68471411243081) } );
    quadraticBeziers.push_back( { float64_t2(-23.791624823334363, 30.68471411243081), float64_t2(-22.973761508142125, 34.6716376952827), float64_t2(-21.4215031922153, 38.283553209017825) } );
    quadraticBeziers.push_back( { float64_t2(-21.4215031922153, 38.283553209017825), float64_t2(-18.336180734516986, 45.46272173454916), float64_t2(-13.001355407288806, 49.99383980822232) } );
    quadraticBeziers.push_back( { float64_t2(-13.001355407288806, 49.99383980822232), float64_t2(-7.771208349095447, 54.43604970005927), float64_t2(-1.5337489636354484, 55.34257906040659) } );
    quadraticBeziers.push_back( { float64_t2(-1.5337489636354484, 55.34257906040659), float64_t2(4.703342661996821, 56.249054969736825), float64_t2(10.57862250293831, 53.421277252750265) } );
    quadraticBeziers.push_back( { float64_t2(-24.645151937487675, 22.22222222222222), float64_t2(-24.645151937487675, 18.726800664983415), float64_t2(-24.07844686432706, 15.307415532017195) } );
    quadraticBeziers.push_back( { float64_t2(-24.07844686432706, 15.307415532017195), float64_t2(-23.535446874511937, 12.03106214740762), float64_t2(-22.493878548799383, 8.95842153372036) } );
    quadraticBeziers.push_back( { float64_t2(-22.493878548799383, 8.95842153372036), float64_t2(-20.433718222207617, 2.880920773303067), float64_t2(-16.71390459269268, -1.726819971507346) } );
    quadraticBeziers.push_back( { float64_t2(-16.71390459269268, -1.726819971507346), float64_t2(-13.068845031967378, -6.24196269997844), float64_t2(-8.365962224765482, -8.677009155076963) } );
    quadraticBeziers.push_back( { float64_t2(-8.365962224765482, -8.677009155076963), float64_t2(-3.664903559470906, -11.111111114560455), float64_t2(1.4064832891699104, -11.111111114560455) } );
    quadraticBeziers.push_back( { float64_t2(1.4064832891699104, -11.111111114560455), float64_t2(6.477870137810726, -11.111111114560455), float64_t2(11.178928803105304, -8.677009155076963) } );
    quadraticBeziers.push_back( { float64_t2(11.178928803105304, -8.677009155076963), float64_t2(15.881811610307198, -6.24196269997844), float64_t2(19.526871171032496, -1.726819971507346) } );
    quadraticBeziers.push_back( { float64_t2(19.526871171032496, -1.726819971507346), float64_t2(23.24668480054744, 2.880920773303067), float64_t2(25.306845127139205, 8.95842153372036) } );
    quadraticBeziers.push_back( { float64_t2(25.306845127139205, 8.95842153372036), float64_t2(26.34841345285176, 12.03106214740762), float64_t2(26.891413442666884, 15.307415532017195) } );
    quadraticBeziers.push_back( { float64_t2(26.891413442666884, 15.307415532017195), float64_t2(27.458118515827497, 18.726800664983415), float64_t2(27.458118515827497, 22.22222222222222) } );
    quadraticBeziers.push_back( { float64_t2(16.480922627682048, 49.40839628516524), float64_t2(21.533002144675336, 44.82221976612453), float64_t2(24.431081709845223, 37.817324246107425) } );
    quadraticBeziers.push_back( { float64_t2(24.431081709845223, 37.817324246107425), float64_t2(25.890133653859575, 34.2906762742334), float64_t2(26.657387466010245, 30.4230279289186) } );
    quadraticBeziers.push_back( { float64_t2(26.657387466010245, 30.4230279289186), float64_t2(27.458118515827497, 26.386624243524338), float64_t2(27.458118515827497, 22.22222222222222) } );
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
    quadraticBeziers.push_back( { float64_t2(46.99684493649464, 22.222222211874193), float64_t2(46.99684493649464, 24.365187357007354), float64_t2(47.50165856850336, 26.40849214254154) } );
    quadraticBeziers.push_back( { float64_t2(46.99684493649464, 22.222222211874193), float64_t2(46.99684493649464, 20.490014270223952), float64_t2(47.32936312964952, 18.81086931184486) } );
    quadraticBeziers.push_back( { float64_t2(47.32936312964952, 18.81086931184486), float64_t2(47.64811789143726, 17.20122672203514), float64_t2(48.25445760890438, 15.733099508064766) } );
    quadraticBeziers.push_back( { float64_t2(48.25445760890438, 15.733099508064766), float64_t2(49.458878172494984, 12.816842248732293), float64_t2(51.55945035137414, 10.904854318747919) } );
    quadraticBeziers.push_back( { float64_t2(51.55945035137414, 10.904854318747919), float64_t2(53.61691320823533, 9.03210545786553), float64_t2(56.10912287718125, 8.51346746225048) } );
    quadraticBeziers.push_back( { float64_t2(56.10912287718125, 8.51346746225048), float64_t2(58.601381647986905, 7.99481924623251), float64_t2(61.01667373080921, 8.936825349788975) } );
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
    quadraticBeziers.push_back( { float64_t2(62.64628497716216, 0), float64_t2(63.0724677849177, -0.7831350486311648), float64_t2(63.73356301236931, -1.2248912826180458) } );
    quadraticBeziers.push_back( { float64_t2(63.73356301236931, -1.2248912826180458), float64_t2(64.39468690728421, -1.6666666708058782), float64_t2(65.14053915599732, -1.6666666708058782) } );
    quadraticBeziers.push_back( { float64_t2(65.14053915599732, -1.6666666708058782), float64_t2(65.88639140471044, -1.6666666708058782), float64_t2(66.54751529962533, -1.2248912826180458) } );
    quadraticBeziers.push_back( { float64_t2(66.54751529962533, -1.2248912826180458), float64_t2(67.20861052707694, -0.7831350486311648), float64_t2(67.63479333483248, 0) } );
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
    quadraticBeziers.push_back( { float64_t2(0, 25.02802262358643), float64_t2(2.717716987840994, 27.27109605998353), float64_t2(4.617684106435822, 30.619741193260307) } );
    quadraticBeziers.push_back( { float64_t2(4.617684106435822, 30.619741193260307), float64_t2(6.517214640646028, 33.967616857477914), float64_t2(7.342404541542, 37.96769858993314) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(-16.829661356280827, 22.222222232570253), float64_t2(-16.829661356280827, 25.299521863322568), float64_t2(-16.20612743942501, 28.27160184927009) } );
    quadraticBeziers.push_back( { float64_t2(-16.20612743942501, 28.27160184927009), float64_t2(-15.608516498153177, 31.120119740565617), float64_t2(-14.475748983946, 33.68878054467064) } );
    quadraticBeziers.push_back( { float64_t2(-14.475748983946, 33.68878054467064), float64_t2(-12.222889570700282, 38.797358958119595), float64_t2(-8.34978002133781, 41.93549692906715) } );
    quadraticBeziers.push_back( { float64_t2(-8.34978002133781, 41.93549692906715), float64_t2(-4.5526499575522985, 45.01207348442188), float64_t2(-0.07304809592371164, 45.47863444658341) } );
    quadraticBeziers.push_back( { float64_t2(-0.07304809592371164, 45.47863444658341), float64_t2(4.406137168208809, 45.94515201946099), float64_t2(8.541846269830518, 43.695269231856976) } );
    quadraticBeziers.push_back( { float64_t2(-16.829661356280827, 22.222222232570253), float64_t2(-16.829661356280827, 19.775427143192953), float64_t2(-16.4329678034509, 17.381857549426734) } );
    quadraticBeziers.push_back( { float64_t2(-16.4329678034509, 17.381857549426734), float64_t2(-16.052867810849897, 15.088410182269635), float64_t2(-15.323769982581526, 12.9375617571727) } );
    quadraticBeziers.push_back( { float64_t2(-15.323769982581526, 12.9375617571727), float64_t2(-13.881657756932702, 8.683311225225527), float64_t2(-11.277788216002662, 5.457892703513304) } );
    quadraticBeziers.push_back( { float64_t2(-11.277788216002662, 5.457892703513304), float64_t2(-8.72624652484287, 2.2972927980676845), float64_t2(-5.434228561958206, 0.5927602794987185) } );
    quadraticBeziers.push_back( { float64_t2(-5.434228561958206, 0.5927602794987185), float64_t2(-2.143487498139084, -1.1111110931745283), float64_t2(1.406483294561572, -1.1111110931745283) } );
    quadraticBeziers.push_back( { float64_t2(1.406483294561572, -1.1111110931745283), float64_t2(4.956454087262228, -1.1111110931745283), float64_t2(8.24719515108135, 0.5927602794987185) } );
    quadraticBeziers.push_back( { float64_t2(8.24719515108135, 0.5927602794987185), float64_t2(11.539213113966012, 2.2972927980676845), float64_t2(14.090754805125808, 5.457892703513304) } );
    quadraticBeziers.push_back( { float64_t2(14.090754805125808, 5.457892703513304), float64_t2(16.694624346055846, 8.683311225225527), float64_t2(18.13673657170467, 12.9375617571727) } );
    quadraticBeziers.push_back( { float64_t2(18.13673657170467, 12.9375617571727), float64_t2(18.865834399973043, 15.088410182269635), float64_t2(19.245934392574046, 17.381857549426734) } );
    quadraticBeziers.push_back( { float64_t2(19.245934392574046, 17.381857549426734), float64_t2(19.642627945403973, 19.775427143192953), float64_t2(19.642627945403973, 22.222222232570253) } );
    quadraticBeziers.push_back( { float64_t2(14.255221756610096, 38.78023588891934), float64_t2(16.785622789519966, 35.56561973980731), float64_t2(18.1833195551059, 31.368235771164848) } );
    quadraticBeziers.push_back( { float64_t2(18.1833195551059, 31.368235771164848), float64_t2(18.890156211195713, 29.245554424684357), float64_t2(19.2583837790249, 26.986851080976148) } );
    quadraticBeziers.push_back( { float64_t2(19.2583837790249, 26.986851080976148), float64_t2(19.642627945403973, 24.629902132545357), float64_t2(19.642627945403973, 22.222222232570253) } );
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
    quadraticBeziers.push_back( { float64_t2(1.846214120639083, 18.936716589248842), float64_t2(5.806231346788918, 19.852176922614927), float64_t2(8.771492365450676, 23.333025865118813) } );
    quadraticBeziers.push_back( { float64_t2(8.771492365450676, 23.333025865118813), float64_t2(10.223941961365416, 25.03802168217522), float64_t2(11.248637775770929, 27.192492534716923) } );
    quadraticBeziers.push_back( { float64_t2(11.248637775770929, 27.192492534716923), float64_t2(12.273729850325969, 29.347796544984533), float64_t2(12.789863693152977, 31.78286227846035) } );
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
    quadraticBeziers.push_back( { float64_t2(-1.1986802262171052, 22.222222215323537), float64_t2(-1.1986802262171052, 22.659284673217268), float64_t2(-1.1106176259184946, 23.081573067853846) } );
    quadraticBeziers.push_back( { float64_t2(-1.1106176259184946, 23.081573067853846), float64_t2(-1.026232375880758, 23.486227376593483), float64_t2(-0.8662337583329167, 23.851594035686166) } );
    quadraticBeziers.push_back( { float64_t2(-0.8662337583329167, 23.851594035686166), float64_t2(-0.5480010468493852, 24.578297952259028), float64_t2(2.6958306655248165e-09, 25.028022623586430) } );
    quadraticBeziers.push_back( { float64_t2(-1.1986802262171052, 22.222222215323537), float64_t2(-1.1986802262171052, 21.834857347938748), float64_t2(-1.1292542160256247, 21.45781551522237) } );
    quadraticBeziers.push_back( { float64_t2(-1.1292542160256247, 21.45781551522237), float64_t2(-1.0627026002354614, 21.0963840768845), float64_t2(-0.9356192259270184, 20.762552410640097) } );
    quadraticBeziers.push_back( { float64_t2(-0.9356192259270184, 20.762552410640097), float64_t2(-0.6837235798543037, 20.100854975343854), float64_t2(-0.2374867735359791, 19.636395894404917) } );
    quadraticBeziers.push_back( { float64_t2(-0.2374867735359791, 19.636395894404917), float64_t2(0.1999377451446994, 19.18110896108879), float64_t2(0.7453526142408801, 18.998013071163935) } );
    quadraticBeziers.push_back( { float64_t2(0.7453526142408801, 18.998013071163935), float64_t2(1.2906004254065488, 18.81497326410479), float64_t2(1.846214120639083, 18.936716589248842) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 0, 25.02802262358643 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
polylines.push_back(polyline);
}
{
CPolyline polyline;
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(51.23730959392052, 33.23475917180379), float64_t2(51.575063246586026, 33.56687499003278), float64_t2(51.93211884462939, 33.86410363018513) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
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
    quadraticBeziers.push_back( { float64_t2(54.6867121621482, 35.50761909810481), float64_t2(56.47036206743867, 36.2032737100014), float64_t2(58.3331992002567, 36.097439878654704) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 51.2373095912247, 33.234759171803788 });
    linePoints.push_back({ 19.119801595116574, 62.00956354193665 });
    linePoints.push_back({ 20.095070888197338, 66.66666666666666 });
    linePoints.push_back({ 67.02893958936777, 66.66666666666666 });
    linePoints.push_back({ 58.3331992002567, 36.097439878654704 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
{
    std::vector<shapes::QuadraticBezier<float64_t>> quadraticBeziers;
    quadraticBeziers.push_back( { float64_t2(48.068592563516596, 44.44444444444444), float64_t2(48.494775371272134, 45.22757949307561), float64_t2(49.155870598723745, 45.669335723613145) } );
    quadraticBeziers.push_back( { float64_t2(49.155870598723745, 45.669335723613145), float64_t2(49.81699449094282, 46.111111108351636), float64_t2(50.56284673965592, 46.111111108351636) } );
    quadraticBeziers.push_back( { float64_t2(50.56284673965592, 46.111111108351636), float64_t2(51.30869898836903, 46.111111108351636), float64_t2(51.969822880588104, 45.669335723613145) } );
    quadraticBeziers.push_back( { float64_t2(51.969822880588104, 45.669335723613145), float64_t2(52.630918108039715, 45.22757949307561), float64_t2(53.05710091579525, 44.44444444444444) } );
    polyline.addQuadBeziers(core::SRange<shapes::QuadraticBezier<float64_t>>(quadraticBeziers.data(), quadraticBeziers.data() + quadraticBeziers.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 54.6867121621482, 35.50761909810481 });
    linePoints.push_back({ 53.049498484610325, 39.999999999310134 });
    linePoints.push_back({ 55.99027074464795, 39.999999999310134 });
    linePoints.push_back({ 55.99027074464795, 44.44444444444444 });
    linePoints.push_back({ 53.05710091579525, 44.44444444444444 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
{
    std::vector<float64_t2> linePoints;
    linePoints.push_back({ 48.068592563516596, 44.44444444444444 });
    polyline.addLinePoints(core::SRange<float64_t2>(linePoints.data(), linePoints.data() + linePoints.size()));
}
polylines.push_back(polyline);
}
