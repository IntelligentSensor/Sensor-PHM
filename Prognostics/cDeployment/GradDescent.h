//
//  GradientDescent.h
//  logistic
//
//  Created by Tunghao on 2019/3/7.
//  Copyright © 2019年 Tunghao. All rights reserved.
//

#ifndef GradDescent_h
#define GradDescent_h

#include <stdio.h>

#define BATCH 24
#define DIMEN 20
#define ITERTION 5

static double target[24]={0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1};

static double Vectors[24][19]={
    {0.11673096854769616, 0.13000017108820472, 0.1409544599990251, 0.13997606237658827, 0.0, 1.0, 0.07024078088991197, 0.10248336838474986, 0.06854614649784611, 0.11343765729557023, 0.09117096108911497, 0.752170439176176, 0.7933158477528032, 0.0, 1.0, 0.7922182716836421, 0.7630673974514961, 0.7425543689938294, 1},
    {0.13353417924944605, 0.13376892552653408, 0.1344433360993661, 0.1340730589960101, 0.0, 1.0, 0.13680939111241155, 0.1382816379591943, 0.1370231706977089, 0.1373603759841249, 0.13797750448971827, 0.7250397622480645, 0.7259542353982683, 0.0, 1.0, 0.7269037511460436, 0.7263631068289972, 0.7232633798440136, 1},
    {0.13348027898758819, 0.13380760302954992, 0.13208700133598728, 0.13305213498306756, 0.0, 1.0, 0.12945608237828615, 0.1294451191016978, 0.13026851728097397, 0.13030140711073906, 0.12856181646510795, 0.7234523896916663, 0.7235260392141772, 0.0, 1.0, 0.7248123738733333, 0.7223464584601336, 0.7232661688829468, 1},
    {0.13191570860122429, 0.13216251567685924, 0.13251704590269017, 0.13302338711645223, 0.0, 1.0, 0.13390971268102958, 0.13393175659989712, 0.13465354411405117, 0.13402675268760092, 0.13477058412062254, 0.7235384081454846, 0.7243203390128562, 0.0, 1.0, 0.724211679437372, 0.7256707822998906, 0.7245231005410745, 1},
    {0.13397644600188177, 0.13430953081301011, 0.13386541773150568, 0.13412811343975378, 0.0, 1.0, 0.1353305637106183, 0.13466439408836164, 0.1335214436882657, 0.13397644600188177, 0.134198502542634, 0.7248989760692215, 0.7255513818539286, 0.0, 1.0, 0.7272361887672507, 0.7262028957192438, 0.7250877553566283, 1},
    {0.15430795925340057, 0.161090342498964, 0.139693954422691, 0.15860781793288636, 0.0, 1.0, 0.08239007876058846, 0.09480270159097863, 0.1268972837053886, 0.14112724064918594, 0.1386447160831083, 0.713510195516926, 0.7303269563361043, 0.0, 1.0, 0.675379414218675, 0.7219349555633278, 0.766966644898628, 1},
    {0.13494959234323042, 0.1371427271657435, 0.13494959234323042, 0.13485728220192383, 0.0, 1.0, 0.13485728220192383, 0.13426058006841002, 0.13391607393099977, 0.1336976657916414, 0.13288255551876912, 0.7207909894853497, 0.729211778086714, 0.0, 1.0, 0.7281433071709291, 0.725442836809367, 0.7267016762521881, 1},
    {0.10900053257994646, 0.11593703454545404, 0.12125377144913715, 0.1271356691011111, 0.0, 1.0, 0.1241350040155914, 0.1144367020026942, 0.12263467147283155, 0.09552954154214174, 0.11232749337562734, 0.6972204941475109, 0.69689669252224, 0.0, 1.0, 0.7466432704657633, 0.7228715299458139, 0.7080598211030361, 1},
    {0.13495209318466642, 0.13498569747519631, 0.13543244059658643, 0.13412010706193664, 0.0, 1.0, 0.1347930759745265, 0.13437093290115656, 0.13424551998154663, 0.13398239411831658, 0.13445044150622654, 0.7242027080913531, 0.7242129874230059, 0.0, 1.0, 0.7242691547356926, 0.7243421182151664, 0.7247229937565136, 1},
    {0.13192824965587477, 0.13271743833466496, 0.1330298300146136, 0.13330036914538007, 0.0, 1.0, 0.14442091197598522, 0.13362397518208327, 0.13283178162545708, 0.13323909276052473, 0.1323856228190432, 0.7394665496634529, 0.7412946699809391, 0.0, 1.0, 0.7431093239400377, 0.7411633026190467, 0.7404854707736149, 1},
    {0.13441587316676326, 0.13426673495816954, 0.13237939120439365, 0.13412266365999312, 0.0, 1.0, 0.1344267908413673, 0.13353624464645283, 0.13318045447415183, 0.13377779116229613, 0.13430455489239895, 0.7244189035536269, 0.7253553203485539, 0.0, 1.0, 0.7238893085948485, 0.7238552776921893, 0.7233132265585239, 1},
    {0.1365430136057034, 0.13722340644093456, 0.1394025562161129, 0.1352450520216108, 0.0, 1.0, 0.1390822742358917, 0.13554233879025113, 0.13592544485684196, 0.13460448806116854, 0.13396392410072622, 0.7224950890391322, 0.7254662041662701, 0.0, 1.0, 0.7241680864457954, 0.7219874407687984, 0.7210053363741672, 1},
    {0.5435533060381558, 0.5240406202648592, 0.19895431832098656, 0.5045279344915445, 0.0, 0.7087106612076348, 0.7691624488215169, 0.3055736161705579, 0.07282233469810044, 1.0, 0.6163756407362379, 0.389129003078942, 0.4000330883217034, 0.25568912254782117, 0.41324147791076754, 0.0, 0.12562297773187978, 1.0, 1},
    {0.0, 0.05358983848623928, 0.8488033871712536, 0.44226497308104545, 0.6732050807568803, 0.1934615859097919, 0.8244016935856268, 0.5755983064143733, 0.6244016935856268, 0.6113248654051938, 1.0, 0.20824616877080881, 0.5616548391159618, 0.4161903906398241, 0.4089061161923614, 0.12330967823191782, 0.0, 1.0, 1},
    {0.4226497308103827, 0.7982940778282492, 0.7264973081037561, 0.6850454237763783, 0.7113248654051914, 0.4848275573014495, 0.0, 0.4737205583711727, 1.0, 0.36047190431930176, 0.13952809568068403, 0.09378359369907711, 0.2753842243057337, 0.16634716571281366, 0.4245233168698993, 0.07985826932957424, 0.0, 1.0, 1},
    {0.21132486540520296, 0.3899576603592808, 0.30064126288223064, 0.26794919243112797, 0.24401693585630563, 0.5358983848622559, 0.47051424396005065, 1.0, 0.0, 0.35726558990817814, 0.6579068527904088, 0.5711808343225476, 0.8668924836554982, 0.0, 0.9623783159996153, 0.593378931155253, 1.0, 0.599548366022328, 1},
    {1.0, 0.9298813283169173, 0.0, 0.6043389585039094, 0.0, 0.6744576301870141, 0.9851821811441946, 0.07241099623768522, 0.11915677735974767, 0.5233728905610312, 0.9127711849064875, 0.6617581346098881, 0.8145330320793959, 0.6143404679884149, 0.3846967115567133, 0.7033044487196962, 0.0, 1.0, 1},
    {0.6077994076022151, 0.4218113852027343, 0.712889239606207, 1.0, 0.758757486000328, 0.4420359328015996, 0.3867814412014083, 0.44203593280161346, 0.5066766696078611, 0.9003293191924124, 0.0, 0.9579560054590396, 0.5146886961749829, 0.6749136980561421, 0.41521153701448993, 1.0, 0.5375160600494282, 0.0, 1},
    {0.31516136112888815, 0.3251384663244349, 0.0, 0.5458754900578006, 0.6948157440666586, 0.48138246824889275, 0.37965438293777043, 1.0, 0.37965438293779596, 0.7220737023733519, 0.41688944644001047, 0.8078777177811004, 0.6535371370315366, 0.25926544655250244, 0.0, 0.4928396130988, 0.3454794404355985, 1.0, 1},
    {0.366025403784467, 0.8564064605510285, 0.3038475772933828, 0.3301270189221995, 0.607695154586741, 0.0, 0.16025403784438666, 0.526279441628829, 1.0, 0.19615242270662953, 0.5621778264910718, 0.0, 0.43054571546352827, 0.1210312574666396, 1.0, 0.2993063685669787, 0.5204161788598157, 0.6673341138242905, 1},
    {0.4255899819273399, 0.27676994578200187, 0.6383849728910098, 0.7093166365455547, 0.896148836678966, 0.6193788908969642, 0.5864593912304751, 1.0, 0.35465831827277733, 0.36161502710900795, 0.0, 0.3163536678550782, 0.19635224009751265, 1.0, 0.0, 0.4764768851460767, 0.32488492996392515, 0.16037179350933126, 1},
    {0.3891835652609886, 0.4432657389560457, 0.42347028949177096, 0.5712255358104619, 0.511839187417624, 0.6793898832005898, 0.4722482884890745, 1.0, 0.0, 0.6108164347390115, 0.4432657389560457, 0.5999202804044228, 0.5042213478947746, 0.7405777370224479, 0.49866959425168916, 1.0, 0.7939789569203594, 0.0, 1},
    {0.5669872981077818, 0.827350269189653, 0.0, 0.2783121635129785, 0.35566243270259834, 0.7113248654051967, 0.345299461620747, 1.0, 0.4226497308103934, 0.4896370289181619, 0.8839745962155702, 0.5222978174664672, 0.5795600545903885, 0.0, 0.8140541388612302, 1.0, 0.3250863019438597, 0.6601624380755681, 1},
    {0.0, 0.2608392471086955, 0.32362096740336477, 1.0, 0.18834516088403275, 0.5313908601473346, 0.7922301072560053, 0.729448386961336, 0.38640268769803404, 0.6666666666666666, 0.9902876340700314, 0.5557274654363007, 0.4075943343242033, 0.9707360434375224, 0.5299973384062702, 1.0, 0.7941125502262932, 0.0, 1}};

static double Vector[24][19]={
    {0.11673097,0.13000017,0.14095446,0.13997606,0.00000000,1.00000000
    ,0.07024078,0.10248337,0.06854615,0.11343766,0.09117096,0.75217044
    ,0.79331585,0.00000000,1.00000000,0.79221827,0.76306740,0.74255437,1},
    {0.13353418,0.13376893,0.13444334,0.13407306,0.00000000,1.00000000
    ,0.13680939,0.13828164,0.13702317,0.13736038,0.13797750,0.72503976
    ,0.72595424,0.00000000,1.00000000,0.72690375,0.72636311,0.72326338,1},
    {0.13348028,0.13380760,0.13208700,0.13305213,0.00000000,1.00000000
    ,0.12945608,0.12944512,0.13026852,0.13030141,0.12856182,0.72345239
    ,0.72352604,0.00000000,1.00000000,0.72481237,0.72234646,0.72326617,1},
    {0.13191571,0.13216252,0.13251705,0.13302339,0.00000000,1.00000000
    ,0.13390971,0.13393176,0.13465354,0.13402675,0.13477058,0.72353841
    ,0.72432034,0.00000000,1.00000000,0.72421168,0.72567078,0.72452310,1},
    {0.13397645,0.13430953,0.13386542,0.13412811,0.00000000,1.00000000
    ,0.13533056,0.13466439,0.13352144,0.13397645,0.13419850,0.72489898
    ,0.72555138,0.00000000,1.00000000,0.72723619,0.72620290,0.72508776,1},
    {0.15430796,0.16109034,0.13969395,0.15860782,0.00000000,1.00000000
    ,0.08239008,0.09480270,0.12689728,0.14112724,0.13864472,0.7135102
    ,0.73032696,0.00000000,1.00000000,0.67537941,0.72193496,0.76696664,1},
    {0.13494959,0.13714273,0.13494959,0.13485728,0.00000000,1.00000000
    ,0.13485728,0.13426058,0.13391607,0.13369767,0.13288256,0.72079099
    ,0.72921178,0.00000000,1.00000000,0.72814331,0.72544284,0.72670168,1},
    {0.10900053,0.11593703,0.12125377,0.12713567,0.00000000,1.00000000
    ,0.12413500,0.11443670,0.12263467,0.09552954,0.11232749,0.69722049
    ,0.69689669,0.00000000,1.00000000,0.74664327,0.72287153,0.70805982,1},
    {0.13495209,0.13498570,0.13543244,0.13412011,0.00000000,1.00000000
    ,0.13479308,0.13437093,0.13424552,0.13398239,0.13445044,0.72420271
    ,0.72421299,0.00000000,1.00000000,0.72426915,0.72434212,0.72472299,1},
    {0.13192825,0.13271744,0.13302983,0.13330037,0.00000000,1.00000000
    ,0.14442091,0.13362398,0.13283178,0.13323909,0.13238562,0.73946655
    ,0.74129467,0.00000000,1.00000000,0.74310932,0.74116330,0.74048547,1},
    {0.13441587,0.13426673,0.13237939,0.13412266,0.00000000,1.00000000
    ,0.13442679,0.13353624,0.13318045,0.13377779,0.13430455,0.7244189
    ,0.72535532,0.00000000,1.00000000,0.72388931,0.72385528,0.72331323,1},
    {0.13654301,0.13722341,0.13940256,0.13524505,0.00000000,1.00000000
    ,0.13908227,0.13554234,0.13592544,0.13460449,0.13396392,0.72249509
    ,0.72546620,0.00000000,1.00000000,0.72416809,0.72198744,0.72100534,1},
    {0.54355331,0.52404062,0.19895432,0.50452793,0.00000000,0.70871066
    ,0.76916245,0.30557362,0.07282233,1.00000000,0.61637564,0.38912900
    ,0.40003309,0.25568912,0.41324148,0.00000000,0.12562298,1.00000000,1},
    {0.00000000,0.05358984,0.84880339,0.44226497,0.67320508,0.19346159
    ,0.82440169,0.57559831,0.62440169,0.61132487,1.00000000,0.20824617
    ,0.56165484,0.41619039,0.40890612,0.12330968,0.00000000,1.00000000,1},
    {0.42264973,0.79829408,0.72649731,0.68504542,0.71132487,0.48482756
    ,0.00000000,0.47372056,1.00000000,0.36047190,0.13952810,0.09378359
    ,0.27538422,0.16634717,0.42452332,0.07985827,0.00000000,1.00000000,1},
    {0.21132487,0.38995766,0.30064126,0.26794919,0.24401694,0.53589838
    ,0.47051424,1.00000000,0.00000000,0.35726559,0.65790685,0.57118083
    ,0.86689248,0.00000000,0.96237832,0.59337893,1.00000000,0.59954837,1},
    {1.00000000,0.92988133,0.00000000,0.60433896,0.00000000,0.67445763
    ,0.98518218,0.07241100,0.11915678,0.52337289,0.91277118,0.66175813
    ,0.81453303,0.61434047,0.38469671,0.70330445,0.00000000,1.00000000,1},
    {0.60779941,0.42181139,0.71288924,1.00000000,0.75875749,0.44203593
    ,0.38678144,0.44203593,0.50667667,0.90032932,0.00000000,0.95795601
    ,0.51468870,0.67491370,0.41521154,1.00000000,0.53751606,0.00000000,1},
    {0.31516136,0.32513847,0.00000000,0.54587549,0.69481574,0.48138247
    ,0.37965438,1.00000000,0.37965438,0.72207370,0.41688945,0.80787772
    ,0.65353714,0.25926545,0.00000000,0.49283961,0.34547944,1.00000000,1},
    {0.36602540,0.85640646,0.30384758,0.33012702,0.60769515,0.00000000
    ,0.16025404,0.52627944,1.00000000,0.19615242,0.56217783,0.00000000
    ,0.43054572,0.12103126,1.00000000,0.29930637,0.52041618,0.66733411,1},
    {0.42558998,0.27676995,0.63838497,0.70931664,0.89614884,0.61937889
    ,0.58645939,1.00000000,0.35465832,0.36161503,0.00000000,0.31635367
    ,0.19635224,1.00000000,0.00000000,0.47647689,0.32488493,0.16037179,1},
    {0.38918357,0.44326574,0.42347029,0.57122554,0.51183919,0.67938988
    ,0.47224829,1.00000000,0.00000000,0.61081643,0.44326574,0.59992028
    ,0.50422135,0.74057774,0.49866959,1.00000000,0.79397896,0.00000000,1},
    {0.56698730,0.82735027,0.00000000,0.27831216,0.35566243,0.71132487
    ,0.34529946,1.00000000,0.42264973,0.48963703,0.88397460,0.52229782
    ,0.57956005,0.00000000,0.81405414,1.00000000,0.32508630,0.66016244,1},
    {0.00000000,0.26083925,0.32362097,1.00000000,0.18834516,0.53139086
    ,0.79223011,0.72944839,0.38640269,0.66666667,0.99028763,0.55572747
    ,0.40759433,0.97073604,0.52999734,1.00000000,0.79411255,0.00000000,1}};


extern void Traning(int Batch, int Dimen, double VectorSet[Batch][Dimen], double TargetSet[], int iteration);

#endif /* GradDescent_h */
